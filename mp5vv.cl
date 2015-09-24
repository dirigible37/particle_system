
#define STEPS_PER_RENDER 30
#define MASS 1.0f
#define DELTA_T (0.002f)
#define FRICTION 0.4f
#define RESTITUTION 0.7f

#define GRAVITY_CONSTANT 1.0f // scale of gravitational pull to sphere
#define SPHERE_RADIUS 1.0f

#define EPS_DOWN (-0.2f) // gravity
#define V_DRAG (4.0f)

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

#define MULT (87.0f)
#define MOD (3647.0f)

float goober(float prev)
{
	prev *= (MOD*MULT);
	return(fmod(prev,MOD)/MOD);
}

__kernel void VVerlet(__global float4* p, __global float4* v, __global float* r, float4 center)
{
	unsigned int i = get_global_id(0);
	float4 force, zoom;
	//float radius, mylength;

	for(int steps=0;steps<STEPS_PER_RENDER;steps++){
		force = getforce(center-p[i],v[i]);
		v[i] += force*DELTA_T/2.0f;
		p[i] += v[i]*DELTA_T;
		force = getforce(center-p[i],v[i]);
		v[i] += force*DELTA_T/2.0f;

        /*
		radius = sqrt(p[i].x*p[i].x + p[i].z*p[i].z);
		if((radius< 0.05f)||(p[i].y<0.0f)){
			// regenerate position and velocity
			zoom.x = r[i]+0.2f;
			r[i] = goober(r[i]);
			zoom.y = 0.2f*r[i]+0.8f;
			r[i] = goober(r[i]);
			zoom.z = 2.0f*(r[i]-0.5f);
			p[i] = zoom;
			v[i] = (float4)(0.0f,0.0f,0.0f,1.0f);
			r[i] = goober(r[i]);
		}
		else{
			// Check for wall collision.  Usually it's (p - q) o n < 0, 
			// where p is the point in question and q is a point in
			// the plane with normal n; but here q = (-.5,0,0) 
			// and n = (1,0,0), so it's simply p.x < -.5.
			if(p[i].x<-0.5){
				// Bounce the point.  Usually it's 
				// vout = vin - (1+r)(vin o n)n
				//      - f*(vin - (vin o n)n)/||vin - (vin o n)n||
				// but here (vin o n)n) = (v[i].x,0,0), and so
				// vin - (vin o n)n = (0,v[i].y,v[i].z).
				p[i].x = -0.49;
				v[i].x = v[i].x - ((1.0 + RESTITUTION)*v[i].x);
				mylength = sqrt(v[i].y*v[i].y + v[i].z*v[i].z);
				if(mylength>0.0f){
					v[i].y -= FRICTION*v[i].y/mylength;
					v[i].z -= FRICTION*v[i].z/mylength;
				}
			}
		}
        */
	}
	//p[i].w = 1.0f;
}
