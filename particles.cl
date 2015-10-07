#define STEPS_PER_RENDER 1
#define MASS 1.0f
#define DELTA_T (0.060f)

#define GRAVITY_CONSTANT 0.3f // scale of gravitational pull to sphere
#define SPHERE_RADIUS 0.25f
#define SPHERE_FRICTION 0.4f
#define SPHERE_RESTITUTION 2.0f

#define WALL_DIST 1.25f
#define CEIL_DIST 1.5f
#define FLOOR_DIST 0.0f
#define PLANE_FRICTION 0.6f
#define PLANE_RESTITUTION 2.0f
#define EPS_DOWN (-0.25f)
#define V_DRAG (2.0f)
#define MULT (87.0f)
#define MOD (3647.0f)

float4 getforce(float4 tocenter, float4 vel)
{
	float4 force;

	force = (GRAVITY_CONSTANT*MASS/dot(tocenter,tocenter)) * normalize(tocenter);
	force -= V_DRAG*vel;
	force.y += EPS_DOWN;

	return(force);
}

void planecollision(__global float4 *p, __global float4 *v, float4 normal, float dist) {
	float p_component = dot(*p, normal);
	if (p_component < dist) {
		*p -= (p_component - dist) * normal;
		float4 zoom = dot(*v, normal) * normal;
		*v -= (1.0f+PLANE_RESTITUTION)*zoom + PLANE_FRICTION*normalize(*v-zoom);
	}
}

float goober(float prev)
{
	prev *= (MOD*MULT);
	return(fmod(prev,MOD)/MOD);
}

__kernel void VVerlet(__global float4* p, __global float4* c, __global float4* v, __global float* r, float4 center)
{
	unsigned int i = get_global_id(0);
	float4 force, normal, zoom;
	float dist;

	for(int steps=0;steps<STEPS_PER_RENDER;steps++){
		force = getforce(center-p[i],v[i]);
		v[i] += force*DELTA_T/2.0f;
		p[i] += v[i]*DELTA_T;
		force = getforce(center-p[i],v[i]);
		v[i] += force*DELTA_T/2.0f;

		normal = p[i] - center;
		dist = length(normal);

		if (dist < SPHERE_RADIUS) {
			normal /= dist;
			p[i] = center + normal*SPHERE_RADIUS;
			dist = dot(v[i], normal);
			if (dist < 0) {
				zoom = dist * normal;
				v[i] -= (1.0f+SPHERE_RESTITUTION)*zoom + SPHERE_FRICTION*normalize(v[i]-zoom);
			}
		}

		planecollision(&p[i], &v[i], (float4)(1.0f,0.0f,0.0f,0.0f), -WALL_DIST);
		planecollision(&p[i], &v[i], (float4)(-1.0f,0.0f,0.0f,0.0f), -WALL_DIST);
		planecollision(&p[i], &v[i], (float4)(0.0f,1.0f,0.0f,0.0f), -FLOOR_DIST);
		planecollision(&p[i], &v[i], (float4)(0.0f,-1.0f,0.0f,0.0f), -CEIL_DIST);
		planecollision(&p[i], &v[i], (float4)(0.0f,0.0f,1.0f,0.0f), -WALL_DIST);
		planecollision(&p[i], &v[i], (float4)(0.0f,0.0f,-1.0f,0.0f), -WALL_DIST);
	}

	c[i].x = (p[i].x+WALL_DIST)/(WALL_DIST*2);
	c[i].y = (p[i].y+FLOOR_DIST)/(FLOOR_DIST+CEIL_DIST);
	c[i].z = (p[i].z+WALL_DIST)/(WALL_DIST*2);
}
