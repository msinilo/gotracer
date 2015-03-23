package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	//"runtime/pprof"
//	"sync/atomic"
	"time"
)

const (
	EPSILON    	= math.SmallestNonzeroFloat32
	RAY_BIAS   	= 0.0005
	RESOLUTION 	= 256
	SPP        	= 12 * 12 // samples per pixel
	MIN_BOUNCES	= 4
	MAX_BOUNCES	= 8

	DIRECT_ILLUMINATION	= true

	DIFFUSE = iota
	GLOSSY
	MIRROR
	MAX_MATERIAL_TYPE
)

type rfloat float64

type Vector struct {
	x, y, z rfloat
}

type Material struct {
	materialType	uint8
	diffuse 		Vector
	emissive        Vector
	specular        Vector
	exp 			rfloat
}

type Sphere struct {
	radius      rfloat
	center      Vector
	material 	*Material

	radiusSqr   rfloat
}

type Ray struct {
	origin Vector
	dir    Vector
}

type Plane struct {
	normal Vector
	d      rfloat
}

type Camera struct {
	right, up, forward 	Vector
	fovScale			rfloat
}

type Scene struct {
	objects []Sphere
	lights  []int
	camera  Camera
}

type FrameBuffer struct {
	w, h int
	buf  []uint32
}

type Context struct {
	channelQuit chan bool
	channelJob  chan WorkChunk
	channelJoin chan bool

	samples [SPP * 2]rfloat

	scene  *Scene
	buffer *FrameBuffer

	progress int32
}

type WorkChunk struct {
	x0, y0, w, h int
}

func Rand01() rfloat {
	return rfloat(rand.Float32())
}

func (mat *Material) IsLight() bool {
	return mat.emissive.x > EPSILON || mat.emissive.y > EPSILON || mat.emissive.z > EPSILON
}

func (scene *Scene) CollectLights() {
	for index, s := range scene.objects {
		if s.material.IsLight() {
			scene.lights = append(scene.lights, index)
		}
	}
}

func (scene *Scene) Initialize() {
	for index := range scene.objects {
		s := &scene.objects[index]
		s.radiusSqr = s.radius * s.radius
	}
}

func (ray *Ray) CalcIntersectionPoint(t rfloat, intersectionPoint *Vector) {
	*intersectionPoint = ray.origin
	Madd(intersectionPoint, intersectionPoint, &ray.dir, t)
}

func (v *Vector) MaxComponent() rfloat {
	maxC := v.x
	if maxC < v.y {
		maxC = v.y
	}
	if maxC < v.z {
		maxC = v.z
	}
	return maxC
}

func InitializeSamplesUniform(samples *[SPP * 2]rfloat) {
	for i := 0; i < SPP*2; i++ {
		samples[i] = Rand01()
	}
}

func InitializeSamples(samples *[SPP * 2]rfloat) {
	xstrata := sqrtf(SPP)
	ystrata := rfloat(SPP) / xstrata

	is := 0
	for ystep := 0; ystep < int(ystrata); ystep++ {
		for xstep := 0; xstep < int(xstrata); xstep++ {
			fx := (rfloat(xstep) + Rand01()) / xstrata
			fy := (rfloat(ystep) + Rand01()) / ystrata
			samples[is] = fx
			samples[is+1] = fy
			is += 2
		}
	}
}

// t, object index
func (scene *Scene) Intersect(ray *Ray) (rfloat, int) {
	var minT rfloat = math.MaxFloat32
	var iSphere = -1
	for i := range scene.objects {
		s := &scene.objects[i]
		t := SphereIntersect(s, ray)
		if t > 0 && t < minT {
			minT = t
			iSphere = i
		}
	}
	if iSphere < 0 {
		minT = -1.0
	}
	return minT, iSphere
}

func Clamp01(x rfloat) rfloat {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func GetColor(color *Vector) (uint32, uint32, uint32) {

	color.x = Clamp01(color.x)
	color.y = Clamp01(color.y)
	color.z = Clamp01(color.z)

	r := uint32(powf(color.x, 0.45)*255.0 + 0.5)
	g := uint32(powf(color.y, 0.45)*255.0 + 0.5)
	b := uint32(powf(color.z, 0.45)*255.0 + 0.5)

	return r, g, b
}

func SampleHemisphere_Uniform(u1, u2 rfloat, dir *Vector) {
	r := sqrtf(1.0 - u1*u1)
	phi := 2.0 * math.Pi * u2

	Set(dir, cosf(phi)*r, sinf(phi)*r, u1)
}

func SampleHemisphere_Cosine(u1, u2 rfloat, dir *Vector) {
	phi := 2.0 * math.Pi * u1
	r := sqrtf(u2)

	sx := cosf(phi) * r
	sy := sinf(phi) * r
	Set(dir, sx, sy, sqrtf(1.0 - sx*sx - sy*sy))
}

func SampleHemisphere_Spec(u1, u2, exp rfloat, dir *Vector) {
	phi := 2.0 * math.Pi * u1
	cosTheta := powf(1.0 - u2, 1.0 / (exp + 1.0))
	sinTheta := sqrtf(1.0 - cosTheta*cosTheta)

	Set(dir, cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta)
}

func fabsf(x rfloat) rfloat {
	return rfloat(math.Abs(float64(x)))
}

func BuildBasis(v1, v2, v3 *Vector) {
	if fabsf(v1.x) > fabsf(v2.y) {
		ooLen := 1.0 / sqrtf(v1.x*v1.x+v1.z*v1.z)
		Set(v2, -v1.z*ooLen, 0.0, v1.x*ooLen)
	} else {
		ooLen := 1.0 / sqrtf(v1.y*v1.y+v1.z*v1.z)
		Set(v2, 0.0, v1.z*ooLen, -v1.y*ooLen)
	}
	Cross(v3, v1, v2)
}

func TransformToBasis(vin, vx, vy, vz, vout *Vector) {
	vout.x = vx.x*vin.x + vy.x*vin.y + vz.x*vin.z
	vout.y = vx.y*vin.x + vy.y*vin.y + vz.y*vin.z
	vout.z = vx.z*vin.x + vy.z*vin.y + vz.z*vin.z
}

func Reflect(dir, n *Vector) Vector {
	h := *n
	Scale(&h, 2.0*Dot(dir, n))
	var reflected Vector
	Sub(&reflected, &h, dir)

	return reflected
}

func SampleLights(scene *Scene, intersectionPoint *Vector, normal, rayDir *Vector, material *Material) Vector {

	result := Vector{0, 0, 0}
	var toLight Vector
	for _, lightIndex := range scene.lights {
		light := scene.objects[lightIndex]
		Sub(&toLight, &light.center, intersectionPoint)
		lightDistSqr := Dot(&toLight, &toLight)
		Normalize(&toLight)

		d := Dot(normal, &toLight)
		if d < 0 {
			d = 0
		}

		var shadowRay Ray
		shadowRay.origin = *intersectionPoint
		shadowRay.dir = toLight

		t, objIndex := scene.Intersect(&shadowRay)
		if t > 0 && objIndex == lightIndex  {

			sinAlphaMaxSqr := light.radiusSqr/lightDistSqr
			cosAlphaMax := sqrtf(1.0 - sinAlphaMaxSqr)

			omega := 2.0 * (1.0 - cosAlphaMax)
			d *= omega

			c := VecMul(&material.diffuse, &light.material.emissive)
			Madd(&result, &result, &c, d)

			// Specular part
			if material.materialType != DIFFUSE {
				reflected := Reflect(&toLight, normal)
				d = -Dot(&reflected, rayDir)
				if d < 0 {
					d = 0.0
				}
				s := material.specular
				smul := powf(d, material.exp)
				Madd(&result, &result, &s, smul)
			}
		}
	}
	return result
}

func InterreflectDiffuse(normal, intersectionPoint *Vector, u1, u2 rfloat, newRay *Ray) {
	var v2, v3 Vector
	BuildBasis(normal, &v2, &v3)

	var sampledDir Vector
	SampleHemisphere_Cosine(u1, u2, &sampledDir)

	newRay.origin = *intersectionPoint
	TransformToBasis(&sampledDir, &v2, &v3, normal, &newRay.dir)
}

func InterreflectSpecular(normal, intersectionPoint *Vector, u1, u2 rfloat, material *Material, newRay *Ray) {
	view := newRay.dir
	Negate(&view)
	reflected := Reflect(&view, normal)
	Normalize(&reflected)

	var v2, v3 Vector
	BuildBasis(&reflected, &v2, &v3)

	var sampledDir Vector
	SampleHemisphere_Spec(u1, u2, material.exp, &sampledDir)

	newRay.origin = *intersectionPoint
	TransformToBasis(&sampledDir, &v2, &v3, &reflected, &newRay.dir)
}

func Trace(ray *Ray, context *Context, u1, u2 rfloat) Vector {

	scene := context.scene
	result := Vector{0, 0, 0}
	rrScale := Vector{1, 1, 1}

	direct := true
	for bounce := 0; bounce < MAX_BOUNCES; bounce++ {

		t, objectIndex := scene.Intersect(ray)

		if t <= 0 {
			break
		}

		hitObject := &scene.objects[objectIndex]
		material := hitObject.material

		if !DIRECT_ILLUMINATION || direct {
			e := VecMul(&rrScale, &material.emissive)
			Add(&result, &result, &e)
		}

		// Russian roulette (termination test)
		diffuse := material.diffuse
		maxDiffuse := diffuse.MaxComponent()
		if bounce >= MIN_BOUNCES || maxDiffuse < EPSILON {
			if Rand01() > maxDiffuse {
				break
			}
			Scale(&diffuse, 1.0 / maxDiffuse)
		}

		var intersectionPoint Vector
		ray.CalcIntersectionPoint(t, &intersectionPoint)

		normal := intersectionPoint
		Sub(&normal, &normal, &hitObject.center)
		Scale(&normal, 1.0 / hitObject.radius)
		if Dot(&normal, &ray.dir) >= 0 {
			Negate(&normal)
		}

		if material.materialType == DIFFUSE {
			direct = false

			if DIRECT_ILLUMINATION {
				directLight := SampleLights(scene, &intersectionPoint, &normal, &ray.dir, material)
				directLight = VecMul(&directLight, &rrScale)
				Add(&result, &result, &directLight)
			}

			InterreflectDiffuse(&normal, &intersectionPoint, u1, u2, ray)
			rrScale = VecMul(&rrScale, &diffuse)
		} else if material.materialType == GLOSSY {

			direct = false

			if DIRECT_ILLUMINATION {
				directLight := SampleLights(scene, &intersectionPoint, &normal, &ray.dir, material)
				directLight = VecMul(&directLight, &rrScale)
				Add(&result, &result, &directLight)
			}

			// Specular/diffuse Russian roulette
			maxSpec := material.specular.MaxComponent()
			p := maxSpec / (maxSpec + maxDiffuse)
			smult := 1.0 / p

			if Rand01() > p { // diffuse
				InterreflectDiffuse(&normal, &intersectionPoint, u1, u2, ray)
				color := diffuse
				Scale(&color, 1.0 / (1.0 - 1.0/smult))	
				rrScale = VecMul(&rrScale, &color)
			} else {
				InterreflectSpecular(&normal, &intersectionPoint, u1, u2, material, ray)
				color := material.specular
				Scale(&color, smult)
				rrScale = VecMul(&rrScale, &color)
			}
		} else {
			view := ray.dir
			Negate(&view)
			reflected := Reflect(&view, &normal)
			Normalize(&reflected)

			ray.origin = intersectionPoint
			ray.dir = reflected

			rrScale = VecMul(&rrScale, &diffuse)
		}

		sampleIdx := rand.Intn(SPP)
		u1 = context.samples[sampleIdx*2]
		u2 = context.samples[sampleIdx*2+1]
	}

	return result
}

func ApplyTentFilter(samples []rfloat, numSamples int) {
	for i := 0; i < numSamples; i++ {
		x := samples[i*2+0]
		y := samples[i*2+1]

		if x < 0.5 {
			samples[i*2] = sqrtf(2.0*x) - 1.0
		} else {
			samples[i*2] = 1.0 - sqrtf(2.0-2.0*x)
		}
		if y < 0.5 {
			samples[i*2+1] = sqrtf(2.0 * y)
		} else {
			samples[i*2+1] = 1.0 - sqrtf(2.0-2.0*y)
		}
	}
}

func ProcessChunk(context *Context, chunk *WorkChunk) {

	endY := chunk.y0 + chunk.h
	endX := chunk.x0 + chunk.w

	var res rfloat = RESOLUTION
	camera := &context.scene.camera

	var viewRay Ray

	cx := Vector{camera.fovScale, 0, 0}
	var cy Vector
	Cross(&cy, &cx, &camera.forward)
	Normalize(&cy)
	Scale(&cy, camera.fovScale)

	invSPP := 1.0 / rfloat(SPP)
	var chunkSamples [SPP * 2]rfloat
	var sphereSamples [SPP * 2]rfloat
	InitializeSamples(&chunkSamples)
	ApplyTentFilter(chunkSamples[0:], SPP)

	for y := chunk.y0; y < endY; y++ {

		yoffset := y * context.buffer.w
		scanline := context.buffer.buf[yoffset:]
		for x := chunk.x0; x < endX; x++ {

			//if x > RESOLUTION / 2 {
			//	InitializeSamplesUniform(&sphereSamples)
			//} else {
				InitializeSamples(&sphereSamples)
			//}

			cr := Vector{0, 0, 0}
			for aa := 0; aa < 4; aa++ {

				aax := rfloat(aa & 0x1)
				aay := rfloat(aa >> 1)

				pr := Vector{0, 0, 0}
				for s := 0; s < SPP; s++ {

					dx := chunkSamples[s*2]
					dy := chunkSamples[s*2+1]

					px := ((aax+0.5+dx)/2.0+rfloat(x))/res - 0.5
					py := -(((aay+0.5+dy)/2.0+rfloat(y))/res - 0.5)

					Set(&viewRay.origin, 50, 52, 295.6)
					Set(&viewRay.dir, 0, 0, 0)

					ccx := cx
					ccy := cy

					Scale(&ccx, px)
					Scale(&ccy, py)
					Add(&viewRay.dir, &ccx, &ccy)
					Add(&viewRay.dir, &viewRay.dir, &camera.forward)
					Normalize(&viewRay.dir)

					//scaledDir := viewRay.dir
					//Scale(&scaledDir, 136.0)
					Madd(&viewRay.origin, &viewRay.origin, &viewRay.dir, 136.0)
					//Add(&viewRay.origin, &viewRay.origin, &scaledDir)

					u1 := sphereSamples[s*2]
					u2 := sphereSamples[s*2+1]
					r := Trace(&viewRay, context, u1, u2)
					Madd(&pr, &pr, &r, invSPP)
				}
				Madd(&cr, &cr, &pr, 0.25)
			}
			r, g, b := GetColor(&cr)

			scanline[x] = 0xFF000000 | (r << 16) | (g << 8) | b
		}
	}

//	progress := atomic.AddInt32(&context.progress, 1)
//	fmt.Printf("%v\n", progress)
}

func workerFunc(ctx *Context) {
	channelJob := ctx.channelJob
	for {
		select {
		case chunk := <-channelJob:
			ProcessChunk(ctx, &chunk)
		case <-ctx.channelQuit:
			ctx.channelJoin <- true
			return
		}
	}
}

func Set(v *Vector, x rfloat, y rfloat, z rfloat) {
	v.x = x
	v.y = y
	v.z = z
}

func Add(result, a, b *Vector) {
	result.x = a.x + b.x
	result.y = a.y + b.y
	result.z = a.z + b.z
}

func Sub(result, a, b *Vector) {
	result.x = a.x - b.x
	result.y = a.y - b.y
	result.z = a.z - b.z
}

func VecMul(a, b *Vector) Vector {
	return Vector{a.x * b.x, a.y * b.y, a.z * b.z}
}

func Scale(v *Vector, s rfloat) {
	v.x *= s
	v.y *= s
	v.z *= s
}

func Negate(v *Vector) {
	v.x = -v.x
	v.y = -v.y
	v.z = -v.z
}

// result = a + b*s
func Madd(result, a, b *Vector, s rfloat) {
	result.x = a.x + b.x * s
	result.y = a.y + b.y * s
	result.z = a.z + b.z * s
}

func Dot(a, b *Vector) rfloat {
	return a.x*b.x + a.y*b.y + a.z*b.z
}

func Cross(result, a, b *Vector) {
	result.x = a.y*b.z - a.z*b.y
	result.y = a.z*b.x - a.x*b.z
	result.z = a.x*b.y - a.y*b.x
}

func Normalize(v *Vector) {
	lenSqr := Dot(v, v)

	if lenSqr > EPSILON {
		len := rfloat(math.Sqrt(float64(lenSqr)))
		v.x /= len
		v.y /= len
		v.z /= len
	}
}

func powf(x, y rfloat) rfloat {
	return rfloat(math.Pow(float64(x), float64(y)))
}

func tanf(x rfloat) rfloat {
	return rfloat(math.Tan(float64(x)))
}

func sqrtf(x rfloat) rfloat {
	return rfloat(math.Sqrt(float64(x)))
}

func sinf(x rfloat) rfloat {
	return rfloat(math.Sin(float64(x)))
}

func cosf(x rfloat) rfloat {
	return rfloat(math.Cos(float64(x)))
}

func SphereIntersect(s *Sphere, r *Ray) rfloat {
	var op Vector
	Sub(&op, &s.center, &r.origin)

	b := Dot(&op, &r.dir)
	d := b*b - Dot(&op, &op) + s.radiusSqr

	if d < 0 {
		return 0
	}

	d = rfloat(math.Sqrt(float64(d)))
	t := b - d

	if t > RAY_BIAS {
		return t
	}

	t = b + d
	if t > RAY_BIAS {
		return t
	}

	return 0
}

func Put16(buffer []uint8, v uint16) {
	buffer[0] = (uint8)(v & 0xFF)
	buffer[1] = (uint8)(v >> 8)
}

func WriteTGAHeader(f *os.File, width uint16, height uint16) {
	var header [18]uint8

	header[2] = 2 // 32-bit
	Put16(header[12:], width)
	Put16(header[14:], height)
	header[16] = 32   // BPP
	header[17] = 0x20 // top down, non interlaced

	binary.Write(f, binary.LittleEndian, header)
}

func WriteTGA(fname string, pixels []uint32, width uint16, height uint16) {
	f, err := os.Create(fname)
	if err != nil {
		return
	}
	defer f.Close()

	WriteTGAHeader(f, width, height)
	binary.Write(f, binary.LittleEndian, pixels)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	var camera Camera
	Set(&camera.right, 1, 0, 0)
	Set(&camera.up, 0, 1, 0)
	Set(&camera.forward, 0, -0.042612, -1)
	Normalize(&camera.forward)
	camera.fovScale = tanf((55.0 * math.Pi / 180.0) * 0.5)

	var scene Scene

	diffuseGrey := Material{materialType: DIFFUSE, diffuse: Vector{.75, .75, .75}}
	diffuseRed := Material{materialType: DIFFUSE, diffuse: Vector{.95, .15, .15}}
	diffuseBlue := Material{materialType: DIFFUSE, diffuse: Vector{.25, .25, .75}}
	diffuseBlack := Material{materialType: DIFFUSE}
	//glossyGreen := Material{materialType: GLOSSY, diffuse: Vector{0.0, 0.55, 14.0 / 255.0}, specular: Vector{1, 1, 1}, exp: 3.0}
	diffuseGreen := Material{materialType: DIFFUSE, diffuse: Vector{0.0, 0.55, 14.0 / 255.0}}
	diffuseWhite := Material{materialType: DIFFUSE, diffuse: Vector{.99, .99, .99}}
	glossyWhite := Material{materialType: GLOSSY, diffuse: Vector{.3, .05, .05}, specular: Vector{0.69, 0.69, 0.69}, exp: 45.0}
	whiteLight := Material{materialType: DIFFUSE, emissive: Vector{400, 400, 400}}
	mirror := Material{materialType: MIRROR, diffuse: Vector{0.999, 0.999, 0.999}}

	scene.camera = camera
	scene.objects = []Sphere{
		Sphere{1e5, Vector{1e5 + 1, 40.8, 81.6}, &diffuseRed, 0},
		Sphere{1e5, Vector{-1e5 + 99, 40.8, 81.6}, &diffuseBlue, 0},
		Sphere{1e5, Vector{50, 40.8, 1e5}, &diffuseGrey, 0},       
		Sphere{1e5, Vector{50, 40.8, -1e5 + 170}, &diffuseBlack, 0},
		Sphere{1e5, Vector{50, 1e5, 81.6}, &diffuseGrey, 0},
		Sphere{1e5, Vector{50, -1e5 + 81.6, 81.6}, &diffuseGrey, 0},
  		Sphere{16.5, Vector{27, 16.5, 57}, &mirror, 0},
		Sphere{10.5, Vector{17, 10.5, 97}, &diffuseGreen, 0},
		Sphere{16.5, Vector{76, 16.5, 78}, &glossyWhite, 0},
		Sphere{8.5, Vector{82, 8.5, 108}, &diffuseWhite, 0},
		//Sphere{600, Vector{50, 681.6-.27, 81.6}, &whiteLight, 0}}
		Sphere{1.5, Vector{50, 81.6 - 16.5, 81.6}, &whiteLight, 0}}
	scene.Initialize()
	scene.CollectLights()

	var fb FrameBuffer
	var context Context

	fb.w = RESOLUTION
	fb.h = RESOLUTION
	fb.buf = make([]uint32, fb.w*fb.h)

	context.buffer = &fb
	context.scene = &scene
	context.channelQuit = make(chan bool)
	context.channelJoin = make(chan bool)
	context.channelJob = make(chan WorkChunk)
	InitializeSamples(&context.samples)

	chunkSize := 16
	chunksPerLine := RESOLUTION / chunkSize
	numWorkers := chunksPerLine * chunksPerLine
	fmt.Printf("Num workers: %v, GOMAXPROCS=%v\n", numWorkers, runtime.GOMAXPROCS(0))

	tstart := time.Now()

	for w := 0; w < numWorkers; w++ {
		go workerFunc(&context)
	}

/*
	    f, err := os.Create("trace.prof")
	      	if err != nil {
	       	//log.Fatal(err)
	      	}
	       pprof.StartCPUProfile(f)
	       defer pprof.StopCPUProfile()
*/
	chunkW := chunkSize
	chunkH := chunkSize

	for y := 0; y < fb.h; y += chunkH {
		for x := 0; x < fb.w; x += chunkW {
			context.channelJob <- WorkChunk{x, y, chunkW, chunkH}
		}
	}

	for w := 0; w < numWorkers; w++ {
		context.channelQuit <- true
	}

	for w := 0; w < numWorkers; w++ {
		<-context.channelJoin
	}

	duration := time.Since(tstart)
	fmt.Printf("Chunk: %v, Took %s\n", chunkSize, duration)

	WriteTGA("test.tga", fb.buf, uint16(fb.w), uint16(fb.h))
}
