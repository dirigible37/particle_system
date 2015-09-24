
// This particle sytem with collisions uses velocity Verlet integration.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include "RGU.h"

#define YAXIS 1
#define BOX 2
#define WALL 3
#define BORDER 4

GLuint OGL_VBO = 1;
#define NUMBER_OF_PARTICLES 512*512
#define DATA_SIZE (NUMBER_OF_PARTICLES*4*sizeof(float)) 

cl_context mycontext;
cl_command_queue mycommandqueue;
cl_kernel mykernel;
cl_program myprogram;
cl_mem oclvbo, dev_velocity, dev_rseed;
size_t worksize[] = {NUMBER_OF_PARTICLES}; 
size_t lws[] = {128}; 

float host_position[NUMBER_OF_PARTICLES][4];
float host_velocity[NUMBER_OF_PARTICLES][4];
float host_rseed[NUMBER_OF_PARTICLES];

void do_kernel()
{
	cl_event waitlist[1];

	clEnqueueNDRangeKernel(mycommandqueue,mykernel,1,NULL,worksize,lws,0,0,
			&waitlist[0]);
	clWaitForEvents(1,waitlist);
}

void do_material()
{
	float mat_ambient[] = {0.0,0.0,0.0,1.0};
	float mat_diffuse[] = {0.0,0.4,0.2,1.0};
	float mat_specular[] = {1.0,1.0,1.0,1.0};
	float mat_shininess[] = {2.0};

	glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
	glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
	glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
	glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
}

void do_material_points()
{
	float mat_ambient[] = {0.0,0.0,0.0,1.0};
	float mat_diffuse[] = {1.0,1.0,0.1,1.0};
	float mat_specular[] = {1.0,1.0,1.0,1.0};
	float mat_shininess[] = {2.0};

	glMaterialfv(GL_FRONT,GL_AMBIENT,mat_ambient);
	glMaterialfv(GL_FRONT,GL_DIFFUSE,mat_diffuse);
	glMaterialfv(GL_FRONT,GL_SPECULAR,mat_specular);
	glMaterialfv(GL_FRONT,GL_SHININESS,mat_shininess);
}

void mydisplayfunc()
{
	void *ptr;
	glFinish();
	clEnqueueAcquireGLObjects(mycommandqueue,1,&oclvbo,0,0,0);
	do_kernel();
	clEnqueueReleaseGLObjects(mycommandqueue, 1, &oclvbo, 0,0,0);
	clFinish(mycommandqueue);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	do_material();
	glCallList(WALL);

	// Frame is drawn to blend with background.
	glEnable(GL_BLEND);
	glDisable(GL_LIGHTING);
	glCallList(YAXIS);
	glCallList(BOX);
	glPushMatrix();
	glTranslatef(-1.0,0.0,0.0);
	glCallList(BOX);
	glTranslatef(0.0,0.0,-1.0);
	glCallList(BOX);
	glTranslatef(1.0,0.0,0.0);
	glCallList(BOX);
	glPopMatrix();
	glDisable(GL_DEPTH_TEST);
	glCallList(BORDER);
	glEnable(GL_DEPTH_TEST);

	glDisable(GL_BLEND);
	glEnable(GL_LIGHTING);
	do_material_points();
	glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
	glVertexPointer(4,GL_FLOAT,0,0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, NUMBER_OF_PARTICLES);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
	glutPostRedisplay();
}

void setup_the_viewvol()
{
	float eye[] = {2.5, 1.8, 2.0};
	float view[] = {0.0, 0.0, 0.0};
	float up[] = {0.0, 1.0, 0.0};

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,1.0,0.1,20.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye[0],eye[1],eye[2],view[0],view[1],view[2],up[0],up[1],up[2]);
}

void do_lights()
{
	float light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
	float light_diffuse[] = { 0.8, 0.8, 0.8, 0.0 };
	float light_specular[] = { 1.0, 1.0, 1.0, 0.0 };
	float light_position[] = { 2.0, 2.0, 2.0, 1.0 };
	float light_direction[] = { -1.0, -1.0, -1.0, 1.0};

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,light_ambient);
	glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER,1);

	glLightfv(GL_LIGHT0,GL_AMBIENT,light_ambient);
	glLightfv(GL_LIGHT0,GL_DIFFUSE,light_diffuse);
	glLightfv(GL_LIGHT0,GL_SPECULAR,light_specular);
	glLightf(GL_LIGHT0,GL_SPOT_EXPONENT,1.0);
	glLightf(GL_LIGHT0,GL_SPOT_CUTOFF,180.0);
	glLightf(GL_LIGHT0,GL_CONSTANT_ATTENUATION,0.5);
	glLightf(GL_LIGHT0,GL_LINEAR_ATTENUATION,0.1);
	glLightf(GL_LIGHT0,GL_QUADRATIC_ATTENUATION,0.01);
	glLightfv(GL_LIGHT0,GL_POSITION,light_position);
	glLightfv(GL_LIGHT0,GL_SPOT_DIRECTION,light_direction);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
}

void build_call_lists()
{

	glNewList(YAXIS,GL_COMPILE);
	glBegin(GL_LINES);
	glColor4f(1.0,1.0,1.0,0.9);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(0.0,1.0,0.0);
	glEnd();
	glEndList();
	glNewList(BOX,GL_COMPILE);
	glBegin(GL_LINES);
	glColor4f(1.0,1.0,1.0,0.9);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(1.0,0.0,0.0);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(0.0,0.0,1.0);
	// Complete box in y=0 plane. 
	glVertex3f(0.0,0.0,1.0);
	glVertex3f(1.0,0.0,1.0);
	glVertex3f(1.0,0.0,0.0);
	glVertex3f(1.0,0.0,1.0);
	glEnd();
	glEndList();
	glNewList(WALL,GL_COMPILE);
	glBegin(GL_QUADS);
	glNormal3f(1.0,0.0,0.0);
	glVertex3f(-0.5,0.0,1.0);
	glVertex3f(-0.5,1.0,1.0);
	glVertex3f(-0.5,1.0,-1.0);
	glVertex3f(-0.5,0.0,-1.0);
	glEnd();
	glEndList();
	glNewList(BORDER,GL_COMPILE);
	glBegin(GL_LINES);
	glColor4f(1.0,1.0,1.0,0.8);
	glVertex3f(-0.5,0.0,-1.0);
	glVertex3f(-0.5,1.0,-1.0);
	glVertex3f(-0.5,1.0,-1.0);
	glVertex3f(-0.5,1.0,1.0);
	glVertex3f(-0.5,1.0,1.0);
	glVertex3f(-0.5,0.0,1.0);
	glEnd();
	glEndList();
}

void InitGL(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE);
	glutInitWindowSize(768,768);
	glutInitWindowPosition(100,50);
	glutCreateWindow("my_cool_cube");
	setup_the_viewvol();
	do_lights();
	glPointSize(1.0);
	glLineWidth(3.0);
	glEnable(GL_LINE_SMOOTH);
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.1,0.2,0.35,1.0);
	build_call_lists();
	glewInit();
	return;
}

double genrand()
{
	return(((double)(random()+1))/2147483649.);
}

void init_particles()
{
	int i, j;
	for(i=0;i<NUMBER_OF_PARTICLES;i++){
		host_position[i][0] = 2.0*(genrand()-0.5);//genrand();
		host_position[i][1] = 2.0*(genrand()-0.5);//0.2*genrand()+0.8;
		host_position[i][2] = 2.0*(genrand()-0.5);
		host_position[i][3] = 1.0;
		for(j=0;j<4;j++) host_velocity[i][j] = 0.0;
		host_rseed[i] = genrand();
	}
}

float center[4] = {0.0, 0.0, 0.0, 1.0};

void InitCL()
{
	cl_platform_id myplatform;
	cl_device_id *mydevice;
	cl_int err;
	char* oclsource; 
	size_t program_length;
	unsigned int gpudevcount;

	err = RGUGetPlatformID(&myplatform);

	err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);
	mydevice = new cl_device_id[gpudevcount];
	err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

	// You need all these to get full interoperability with OpenGL:
	cl_context_properties props[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)myplatform,
		0};

	mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
	mycommandqueue = clCreateCommandQueue(mycontext,mydevice[0],0,&err);

	oclsource = RGULoadProgSource("mp5vv.cl", "", &program_length);
	myprogram = clCreateProgramWithSource(mycontext,1,(const char **)&oclsource,
			&program_length, &err);
	if(err==CL_SUCCESS) fprintf(stderr,"create ok\n");
	else fprintf(stderr,"create err %d\n",err);
	clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
	mykernel = clCreateKernel(myprogram, "VVerlet", &err);
	if(err==CL_SUCCESS) fprintf(stderr,"build ok\n");
	else {
        fprintf(stderr,"build err %d\n",err);
        char log[512];
        clGetProgramBuildInfo(myprogram,mydevice[0],CL_PROGRAM_BUILD_LOG,sizeof(log),log,NULL);
	    printf("%s", log);
    }

	glBindBuffer(GL_ARRAY_BUFFER, OGL_VBO);
	glBufferData(GL_ARRAY_BUFFER, DATA_SIZE, &host_position[0][0], GL_DYNAMIC_DRAW);
	oclvbo = clCreateFromGLBuffer(mycontext,CL_MEM_WRITE_ONLY,OGL_VBO,&err);

	dev_velocity = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
			DATA_SIZE,&host_velocity[0][0],&err); 

	dev_rseed = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
			NUMBER_OF_PARTICLES*sizeof(float),&host_rseed[0],&err); 

	clSetKernelArg(mykernel,0,sizeof(cl_mem),&oclvbo);
	clSetKernelArg(mykernel,1,sizeof(cl_mem),&dev_velocity);
	clSetKernelArg(mykernel,2,sizeof(cl_mem),&dev_rseed);
	clSetKernelArg(mykernel,3,sizeof(float)*4,center);
}

void cleanup()
{
	clReleaseKernel(mykernel);
	clReleaseProgram(myprogram);
	clReleaseCommandQueue(mycommandqueue);
	glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
	glDeleteBuffers(1,&OGL_VBO);
	clReleaseMemObject(oclvbo);
	clReleaseMemObject(dev_velocity);
	clReleaseMemObject(dev_rseed);
	clReleaseContext(mycontext);
	exit(0);
}

void getout(unsigned char key, int x, int y)
{
	switch(key) {
		case 'q':
			cleanup();
		default:
			break;
	}
}

int main(int argc,char **argv)
{
	srandom(123456789);
	init_particles();
	InitGL(argc, argv); 
	InitCL(); 
	glutDisplayFunc(mydisplayfunc);
	glutKeyboardFunc(getout);
	glutMainLoop();
}
