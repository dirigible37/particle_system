
#define STEPS_PER_RENDER 20
#define MASS 1.0f
#define DELTA_T (0.003f)

#define GRAVITY_CONSTANT 0.3f // scale of gravitational pull to sphere
#define SPHERE_RADIUS 0.25f
#define SPHERE_FRICTION 0.4f
#define SPHERE_RESTITUTION 2.0f

#define WALL_DIST 1.25f
#define CEIL_DIST 1.0f
#define FLOOR_DIST 0.0f
#define PLANE_FRICTION 0.6f
#define PLANE_RESTITUTION 2.0f

#define EPS_DOWN (-0.25f) // gravity
#define V_DRAG (2.0f)

//
// Use the vector to the center of the sphere to calculate gravitational force,
// then remove drag and the gravitational pull of the earth
//
float4 getforce(float4 tocenter, float4 vel)
{
    float4 force;

    // Calculate gravitational force based on vector to center of sphere
    force = (GRAVITY_CONSTANT*MASS/dot(tocenter,tocenter)) * normalize(tocenter);
    // Take away drag
    force -= V_DRAG*vel;
    // Add downward gravity
    force.y += EPS_DOWN;
    
	return(force);
}

//
// Make a particle bounce off a plane, given the plane's normal and distance from origin
// Distance from origin may be negative, meaning the normal points towards the origin
//
void planecollision(__global float4 *p, __global float4 *v, float4 normal, float dist) {
    float p_component = dot(*p, normal);
    if (p_component < dist) {
        // Force particle to surface
        *p -= (p_component - dist) * normal;
        // Bounce it out with friction
        float4 zoom = dot(*v, normal) * normal;
        *v -= (1.0f+PLANE_RESTITUTION)*zoom + PLANE_FRICTION*normalize(*v-zoom);
    }
}

#define MULT (87.0f)
#define MOD (3647.0f)

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
        
        // Check for collision with sphere
        normal = p[i] - center;
        dist = length(normal);

        if (dist < SPHERE_RADIUS) {
            // Normalize normal and move point to outside of sphere
            normal /= dist;
            p[i] = center + normal*SPHERE_RADIUS;
            // Get the component of the velocity in the direction of the normal
            dist = dot(v[i], normal);
            // Since the sphere is moving, make sure the particle is actually moving into it
            if (dist < 0) {
                zoom = dist * normal;
                // Bounce it out with friction
                v[i] -= (1.0f+SPHERE_RESTITUTION)*zoom + SPHERE_FRICTION*normalize(v[i]-zoom);
            }
        }

        // Collide with walls
        planecollision(&p[i], &v[i], (float4)(1.0f,0.0f,0.0f,0.0f), -WALL_DIST);
        planecollision(&p[i], &v[i], (float4)(-1.0f,0.0f,0.0f,0.0f), -WALL_DIST);
        planecollision(&p[i], &v[i], (float4)(0.0f,1.0f,0.0f,0.0f), -FLOOR_DIST);
        planecollision(&p[i], &v[i], (float4)(0.0f,-1.0f,0.0f,0.0f), -CEIL_DIST);
        planecollision(&p[i], &v[i], (float4)(0.0f,0.0f,1.0f,0.0f), -WALL_DIST);
        planecollision(&p[i], &v[i], (float4)(0.0f,0.0f,-1.0f,0.0f), -WALL_DIST);
	}

    // Also set color or something
    c[i].x = floor(((p[i].x+1.25)/2.5)*3)/2;
    c[i].y = floor(p[i].y*3)/2;
    c[i].z = floor(((p[i].z+1.25)/2.5)*3)/2;
}
