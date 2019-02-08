/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Particle system example with collisions using uniform grid

    CUDA 2.1 SDK release 12/2008
    - removed atomic grid method, some optimization, added demo mode.

    CUDA 2.2 release 3/2009
    - replaced sort function with latest radix sort, now disables v-sync.
    - added support for automated testing and comparison to a reference value.
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA runtime
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h> // includes cuda_gl_interop.h// includes cuda_gl_interop.h

// Includes
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>

#include "particleSystem.h"
#include "render_particles.h"
#include "paramgl.h"

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f
#define CUDART_PI_F         3.141592654f

#define GRID_SIZE       64
#define NUM_PARTICLES 36864;//18432;// //9600; //18432
int maxStepCount =2*320000;//6000000*2.0f;//16000000;

const uint width = 800, height = 600;

// view params
int ox, oy;
int buttonState = 0;
float camera_trans[] = {0, 0, -3};
float camera_rot[]   = {0, 0, 0};
float camera_trans_lag[] = {0, 0, -3};
float camera_rot_lag[] = {0, 0, 0};
const float inertia = 0.1f;
ParticleRenderer::DisplayMode displayMode = ParticleRenderer::PARTICLE_SPHERES;

int mode = 0;
bool displayEnabled = true;
bool bPause = false;
bool systemReady = true;
bool endMessage = true;
bool displaySliders = false;
bool wireframe = false;
bool demoMode = false;
bool video = false;
bool output = false;
bool printCam = false;
bool grid = false;

unsigned char *g_video_memory_start = NULL;
unsigned char *g_video_memory_ptr = NULL;
int g_video_seconds_total = 2;
int g_video_fps = 30;

int idleCounter = 0;
int demoCounter = 0;
const int idleDelay = 2000;

enum { M_VIEW = 0};

//Restitution coefficient
float e = 0.9f;//0.6
float et = 0.0f;

//Binary collision time
float tc = 1e-3f;//0.5e-3f;//0.125                  /////////////////////////////////////////////

uint numParticles = 0;
uint3 gridSize;
uint3 feedspacing;
float gapThickness = 0.0176f;
float3 initVelocity;
float maxDropHeight;
int numIterations = 0; // run until exit

// simulation parameters
float timestep = tc/40.0f;// 2.5e-5f;
float damping = 1.0f;
float gravity = 9.81f;
int iterations = 1;
uint recordStep = 0.01f/timestep+1.0f;// 4000;

/*********
DEM simulation parameters for linear-spring dashpot force model for particle-particle collisions
**********/

//Normal spring stiffness coefficient
float collideNSpring = powf((CUDART_PI_F/tc),2.0f)+powf((log(e)/tc),2.0f);



//Tangential spring stiffness coefficient
float collideTSpring = 2.0f / 7.0f * collideNSpring;
//Tangential velocity damping term
float collideShear = 0.0f;
//Normal velocity based damping coefficient
float collideNDamping = -log(e)/tc;
//Tangential velocity based damping coefficient
//float collideTDamping = 2.0f * sqrt(collideTSpring)*log(1/et)/(sqrt(powf(CUDART_PI_F,2.0f)+powf(log(1/et),2.0f)));
float collideTDamping = 2.0f / 7.0f * collideNDamping;
//float collideTDamping = 0.0f;
//Coloumb coefficient of sliding friction for use in tangential force model
float collideColoumb = 0.2f;
//Attractive force coefficient for modeling cohesive grains
float collideAttraction = 0.0f;

/*********
DEM simulation parameters for particle-boundary wall collisions
**********/

//Normal spring stiffness coefficient
float boundarySpring = powf((CUDART_PI_F/tc),2.0f)+powf((log(e)/tc),2.0f);
//Normal velocity based damping term
float boundaryDamping = -log(e)/tc;
//Coloumb coefficient of sliding friction for use in tangential force model
float boundaryColoumb = 0.0f;
//Tangential velocity based damping term
float boundaryShear = 0.0f;

float smallParticleRadius = 0.002f; //density bi
float largeParticleRadius = smallParticleRadius*1.0001; //density bi

//float smallParticleRadius = 0.001f;// 0.001f; //size bi
//float largeParticleRadius = 0.002;//2.0f*smallParticleRadius; //size bi
//float smallParticleRadius = largeParticleRadius/2.0f;// 0.001f; //size bi
//Rolling friction coefficient
//float collideRollingFriction = 2*largeParticleRadius * 0.045f;
float collideRollingFriction = 0.0f;
//float boundaryRollingFriction = 2*largeParticleRadius * 0.045f;
float boundaryRollingFriction = 0.0f;

ParticleSystem *psystem = 0;

// fps
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;
StopWatchInterface *kernelTime = NULL;

ParticleRenderer *renderer = 0;

float modelView[16];
float windowAttributes[4];

ParamListGL *params;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
char        *g_refFile = NULL;
char		*g_restartFile = NULL;
std::ifstream	inputFile;
std::ifstream	restartFile;
float *numberParticles = new float[1];
int number = 0;
long long int lastPos = 0;
long long int secondToLastPos = 0;
long long int frame = 0;
float *nextNumParticles = new float[1];

// For restart only
float *x = new float[1];
float *y = new float[1];
float *z = new float[1];
float *r = new float[1];
float *type = new float[1];
float *i = new float[1];
float *vx = new float[1];
float *vy = new float[1];
float *vz = new float[1];

float *pos;
float *vel;
float *mass;
float *col;
float *omegvel;

const char *sSDKsample = "CUDA Particles Simulation";

extern "C" void cudaInit(int argc, char **argv);
extern "C" void cudaGLInit(int argc, char **argv);
extern "C" void copyArrayFromDevice(void *host, const void *device, unsigned int vbo, int size);
void frameDump();
bool writeFrame();
bool writeFrames();
void drawGrid();
void runTest(int numIter);
void computeFPS();

// initialize particle system
void initParticleSystem(int numParticles, uint3 gridSize, bool bUseOpenGL)
{

    psystem = new ParticleSystem(numParticles, gridSize, bUseOpenGL, smallParticleRadius, largeParticleRadius, gapThickness);
	psystem->setRecordStep(recordStep);
    psystem->reset(ParticleSystem::CONFIG_GRID);
	//bPause = true;
    printf("Particle system reset success\n");
    printf("collideNSpring %f\n",collideNSpring);
opengl:
    if (bUseOpenGL)
    {
        renderer = new ParticleRenderer;
        renderer->setParticleRadius(psystem->getParticleRadius());
        renderer->setColorBuffer(psystem->getColorBuffer());
		renderer->setVelocityBuffer(psystem->getVelocityBuffer());
        renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getCurrentTotal());
    }

    sdkCreateTimer(&timer);
	sdkCreateTimer(&kernelTime);
	sdkStartTimer(&kernelTime);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
	sdkDeleteTimer(&kernelTime);
}

// initialize OpenGL
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("cuDEM");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object"))
    {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

#if defined (_WIN32)

    if (wglewIsSupported("WGL_EXT_swap_control"))
    {
        // disable vertical sync
        wglSwapIntervalEXT(0);
    }

#endif

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);  //Background window color

    glutReportErrors();
}

void runBenchmark(int iterations, char *exec_path)
{
    printf("be Run %u particles simulation for %d iterations...\n\n", numParticles, iterations);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);

    for (int i = 0; i < iterations; ++i)
    {
        psystem->update(timestep);
    }

    cudaDeviceSynchronize();
    sdkStopTimer(&timer);
    float fAvgSeconds = ((float)1.0e-3 * (float)sdkGetTimerValue(&timer)/(float)iterations);

    printf("particles, Throughput = %.4f KParticles/s, Time = %.5f s, Size = %u particles, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-3 * numParticles)/fAvgSeconds, fAvgSeconds, numParticles, 1, 0);

    if (g_refFile)
    {
        printf("\nChecking result...\n\n");
        float *hPos = (float *)malloc(sizeof(float)*4*psystem->getNumParticles());
        copyArrayFromDevice(hPos, psystem->getCudaPosVBO(),
                            0, sizeof(float)*4*psystem->getNumParticles());

        sdkDumpBin((void *)hPos, sizeof(float)*4*psystem->getNumParticles(), "particles.bin");

        if (!sdkCompareBin2BinFloat("particles.bin", g_refFile, sizeof(float)*4*psystem->getNumParticles(),
                                    MAX_EPSILON_ERROR, THRESHOLD, exec_path))
        {
            g_TotalErrors++;
        }
    }
}

void runTest(int iterations)
{
	psystem->changeWriteState();

	printf("te Running simulation for %d iterations and writing to output.\n",iterations);

	for (int i = 0; i < iterations; i++)
	{
//		if(i>0.05f/timestep)
//			{psystem->setGravity(100.0f);}      /////////////// packing with gravity , while running without
		psystem->update(timestep);
	}

	std::cout << "Output file generated!\n\nQuitting...\n" << std::endl;

}

void computeFPS()
{
    frameCount++;
    fpsCount++;
    if((psystem->getStepCount() % 400)==0)
    {
    //	std::cout<<"Iterations: "<<psystem->getStepCount()<<" "<<"Walltime: "<<(float)sdkGetTimerValue(&kernelTime)*1e-3f<<std::endl;
    }

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		float ips = psystem->getStepCount() / (sdkGetTimerValue(&kernelTime) * 1.0e-3f); //Iterations per second
		float ktime = (sdkGetTimerValue(&kernelTime) / 60000.0f);
        sprintf(fps, "cuDEM (%d particles): %3.0f iterations/s | Simulated time: %3.5f s | Wall time: %4.0f min", psystem->getCurrentTotal(), ips, psystem->getStepCount() * timestep, ktime);

        if (displayEnabled)
        {
        	glutSetWindowTitle(fps);
        }
        fpsCount = 0;

        fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void printCamera(int x, int y, std::string camPos, std::string camRot)
{
//(x,y) is from the bottom left of the window
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glGetFloatv(GL_VIEWPORT, windowAttributes);
    glOrtho(0, windowAttributes[2], 0, windowAttributes[3], -1.0f, 1.0f);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glPushAttrib(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);
    glColor3f(1.0f,1.0f,1.0f);
    glRasterPos2i(x,y);
    for (int i=0; i<camPos.size(); i++)
    {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, camPos[i]);
    }
    glRasterPos2i(x,y-15);
    for (int i=0; i<camRot.size(); i++)
    {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, camRot[i]);
    }
    glPopAttrib();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

void display()
{
    sdkStartTimer(&timer);

	systemReady = psystem->getStatus();

    // update the simulation
    if (!bPause && systemReady)
    {
        psystem->setIterations(iterations);
        psystem->setDamping(damping);
        psystem->setGravity(-gravity);
        psystem->setCollideNSpring(collideNSpring);
		psystem->setCollideTSpring(collideTSpring);
        psystem->setCollideNDamping(collideNDamping);
        psystem->setCollideTDamping(collideTDamping);
		psystem->setCollideShear(collideShear);
		psystem->setColoumbStaticFric(collideColoumb);
        psystem->setCollideAttraction(collideAttraction);
		psystem->setBoundarySpring(boundarySpring);
		psystem->setBoundaryDamping(boundaryDamping);
		psystem->setBoundaryShear(boundaryShear);
		psystem->setBoundaryColoumb(boundaryColoumb);
		psystem->setCollideRollingFriction(collideRollingFriction);
		psystem->setBoundaryRollingFriction(boundaryRollingFriction);

        psystem->update(timestep);

        if (renderer)
        {
            renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getCurrentTotal());
        }
    }
	if(!systemReady && endMessage)
	{
		std::cout << "Number of particles exceeds maximum number specified. Quitting..." << std::endl;
		endMessage = !endMessage;
		printf("Simulation terminated.");
        exit(EXIT_SUCCESS);
		delete psystem;
		free(g_video_memory_ptr);
		free(g_video_memory_start);
	}

	if(psystem->getStepCount() > maxStepCount)
	{
		std::cout << "Simulation time exceeds maximum prescribed simulation time. Quitting..." << std::endl;
		endMessage = !endMessage;
		printf("Simulation terminated.");
        exit(EXIT_SUCCESS);
		delete psystem;
		free(g_video_memory_ptr);
		free(g_video_memory_start);
	}

    // render
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // view transform
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (int c = 0; c < 3; ++c)
    {
        camera_trans_lag[c] += (camera_trans[c] - camera_trans_lag[c]) * inertia;
        camera_rot_lag[c] += (camera_rot[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(camera_trans_lag[0], camera_trans_lag[1], camera_trans_lag[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);

    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);

    // cube
    //glColor3f(1.0, 1.0, 1.0);
    //glutWireCube(2.0);

    if (renderer && displayEnabled)
    {
        renderer->display(displayMode);
    }

    if (displaySliders)
    {
        glDisable(GL_DEPTH_TEST);
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO); // invert color
        glEnable(GL_BLEND);
        params->Render(0, 0);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
    }

    sdkStopTimer(&timer);

    if (printCam)
    {
		char camPos[64];
		char camRot[64];
		sprintf(camPos, "camera Pos x: %1.3f, y: %1.3f, z: %1.3f",camera_trans[0],camera_trans[1],camera_trans[2]);
		sprintf(camRot, "camera Rot x: %1.3f, y: %1.3f, z: %1.3f",camera_rot[0],camera_rot[1],camera_rot[2]);
		printCamera(40,40,camPos,camRot);
    }

    drawGrid();

    glutSwapBuffers();

	if (video) 
	{
		frameDump();
	}

    glutReportErrors();

	computeFPS();
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}

void feedSwitch()
{
	bool isOn = psystem->getFeedOn();
	psystem->setFeedOn(isOn);
}

void addParticles()
{
	// Dynamically add particles from feed
	float pd = 0.0035f;
	float j = psystem->getParticleRadius()*0.01f;
	uint gridSize[3];
	gridSize[0] = 6; gridSize[1] = 1; gridSize[2] = 16;
	uint number = gridSize[0] * gridSize[1] * gridSize[2];

	//psystem->addParticles(gridSize,pd,j,number);
}

void frameDump()
{
	if (psystem->getStepCount() % uint((1.0f / timestep / g_video_fps)) == 0)
	{
		glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, g_video_memory_ptr);
		g_video_memory_ptr += width * height * 3;
	}
}

bool writeFrames()
{
	g_video_memory_ptr = g_video_memory_start;
	for (int i = 0; i < g_video_seconds_total * g_video_fps; i++)
	{
		if (!writeFrame())
		{
			return false;
		}
		g_video_memory_ptr += width * height * 3;
	}
	free(g_video_memory_start);
	printf("Video images dumped\n");
	return true;
}

bool writeFrame()
{
	static long int frame_number = 0;
	// write into a file
	char name[1024];
	// save name will have number
	sprintf(name, "video_frame_%03d.raw", frame_number);
	std::ofstream file;
	file.open(name, std::ios::out | std::ios::binary);
	if (!file.is_open())
	{
		printf("ERROR: writing video frame image. Could not open %s for writing\n", name);
		return false;
	}
	// natural order is upside down so flip y
	int bytes_in_row = width * 3;
	int bytes_left = width * height * 3;
	while (bytes_left > 0)
	{
		int start_of_row = bytes_left - bytes_in_row;
		// write the row
		for (int i = 0; i < bytes_in_row; i++)
		{
			file << g_video_memory_ptr[start_of_row + i];
		}
		bytes_left -= bytes_in_row;
	}
	file.close();
	// invoke ImageMagick to convert from .raw to .png
	char command[2056];
	sprintf(command, "convert -depth 8 -size %ix%i rgb:video_frame_%03d.raw video_frame_%03d.png", width, height, frame_number, frame_number);
	printf("%s\n", command);
	system(command);
	//delete the .raw
	sprintf(command, "del %s", name);
	system(command);

	frame_number++;
	return true;
}

void writeBin()
{
	psystem->changeWriteState();
}


void reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.001, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glViewport(0, 0, w, h);

    if (renderer)
    {
        renderer->setWindowSize(w, h);
        renderer->setFOV(60.0);
    }
}

void mouse(int button, int state, int x, int y)
{
    int mods;

    if (state == GLUT_DOWN)
    {
        buttonState |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT)
    {
        buttonState = 2;
    }
    else if (mods & GLUT_ACTIVE_CTRL)
    {
        buttonState = 3;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    if (displaySliders)
    {
        if (params->Mouse(x, y, button, state))
        {
            glutPostRedisplay();
            return;
        }
    }

    glutPostRedisplay();
}

// transfrom vector by matrix
void xform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[4] + v[2]*m[8] + m[12];
    r[1] = v[0]*m[1] + v[1]*m[5] + v[2]*m[9] + m[13];
    r[2] = v[0]*m[2] + v[1]*m[6] + v[2]*m[10] + m[14];
}

// transform vector by transpose of matrix
void ixform(float *v, float *r, GLfloat *m)
{
    r[0] = v[0]*m[0] + v[1]*m[1] + v[2]*m[2];
    r[1] = v[0]*m[4] + v[1]*m[5] + v[2]*m[6];
    r[2] = v[0]*m[8] + v[1]*m[9] + v[2]*m[10];
}

void ixformPoint(float *v, float *r, GLfloat *m)
{
    float x[4];
    x[0] = v[0] - m[12];
    x[1] = v[1] - m[13];
    x[2] = v[2] - m[14];
    x[3] = 1.0f;
    ixform(x, r, m);
}

void drawGrid()
{
	if(grid)
	{
	uint3 gridSize = psystem->getGridSize();
	float3 cellSize = psystem->getCellSize();
	float3 worldO = psystem->getWorldOrigin();

	GLfloat zEnd = gridSize.z*cellSize.z+worldO.z;
	GLfloat xEnd = gridSize.x*cellSize.x+worldO.x;
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glBegin(GL_LINES);
	for(int i=0;i<gridSize.y;i++)
	{
		for(int j=0;j<gridSize.x;j++)
		{
			glColor4f(0.5,0.5,0.5,0.5);
			glVertex3f(j*cellSize.x+worldO.x,i*cellSize.y+worldO.y,worldO.z);
			glVertex3f(j*cellSize.x+worldO.x,i*cellSize.y+worldO.y,zEnd);
		}

	}
	glEnd();
	glBegin(GL_LINES);
	for(int i=0;i<gridSize.y;i++)
	{
		for(int j=0;j<gridSize.z;j++)
		{
			glColor4f(0.5,0.5,0.5,0.5);
			glVertex3f(worldO.x,i*cellSize.y+worldO.y,j*cellSize.z+worldO.z);
			glVertex3f(xEnd,i*cellSize.y+worldO.y,j*cellSize.z+worldO.z);
		}

	}
	glEnd();
	}
	glDisable(GL_BLEND);
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (displaySliders)
    {
        if (params->Motion(x, y))
        {
            ox = x;
            oy = y;
            glutPostRedisplay();
            return;
        }
    }

    switch (mode)
    {
        case M_VIEW:
            if (buttonState == 3)
            {
                // left+middle = zoom
                camera_trans[2] += (dy / 100.0f) * 0.5f * fabs(camera_trans[2]);
            }
            else if (buttonState & 2)
            {
                // middle = translate
                camera_trans[0] += dx / 100.0f;
                camera_trans[1] -= dy / 100.0f;
            }
            else if (buttonState & 1)
            {
                // left = rotate
                camera_rot[0] += dy / 5.0f;
                camera_rot[1] += dx / 5.0f;
            }

            break;
    }

    ox = x;
    oy = y;

    demoMode = false;
    idleCounter = 0;

    glutPostRedisplay();
}
void spacemotion(int x, int y, int z)
{

	camera_trans[0]+=x/15000.0f;
	camera_trans[1]+=y/150000.0f;
	camera_trans[2]-=z/30000.0f;

    glutPostRedisplay();
}

void spacerotate(int x, int y, int z)
{

    camera_rot[0] -= x/450.0f;
    camera_rot[1] -= y/150.0f;
    camera_rot[2] -= z/150.0f;

	glutPostRedisplay();
}

// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case ' ':
            bPause = !bPause;
            break;

        case 13:
            psystem->update(timestep);

            if (renderer)
            {
                renderer->setVertexBuffer(psystem->getCurrentReadBuffer(), psystem->getCurrentTotal());
            }

            break;

        case '\033':
        case 'q':
			printf("Simulation terminated.");
            exit(EXIT_SUCCESS);
			delete psystem;
			free(g_video_memory_ptr);
			free(g_video_memory_start);

            break;

        case 'v':
            mode = M_VIEW;
            break;

        case 'p':
            displayMode = (ParticleRenderer::DisplayMode)
                          ((displayMode + 1) % ParticleRenderer::PARTICLE_NUM_MODES);
            break;

        case 'd':
            psystem->dumpGrid();
            break;

        case 'u':
			psystem->update(timestep);
            psystem->dumpParticles(0, psystem->getCurrentTotal());
            break;

		case 'n':
			psystem->dumpNeighborLists();
			break;

		case 'k':
			video = !video;
			if (video == true)
			{
				printf("Video record enabled\n");
			}
			else
			{
				printf("Video record disabled\n");
				writeFrames();
			}
			break;

        case 'r':
            displayEnabled = !displayEnabled;
            break;

        case '1':
            psystem->reset(ParticleSystem::CONFIG_GRID);
            break;

        case '2':
            psystem->reset(ParticleSystem::CONFIG_RANDOM);
            break;

		case '3':
			addParticles();
			break;

		case '4':
			feedSwitch();
			break;

		case 'o':
			writeBin();
			break;

        case 'w':
            wireframe = !wireframe;
            break;

        case 'h':
            displaySliders = !displaySliders;
            break;

        case 'g':
        	grid = !grid;
        	break;
    }

    demoMode = false;
    idleCounter = 0;
    glutPostRedisplay();
}

void special(int k, int x, int y)
{
    if (displaySliders)
    {
        params->Special(k, x, y);
    }

    demoMode = false;
    idleCounter = 0;
}

void idle(void)
{
    glutPostRedisplay();
}

void initParams(bool test)
{
    if (g_refFile)
    {
        timestep = 0.0f;
        damping = 0.0f;
        gravity = 0.0f;
        collideNSpring = 0.0f;
        collideNDamping = 0.0f;
        collideTDamping = 0.0f;
		collideColoumb = 0.5f;
        collideAttraction = 0.0f;

    }
    if (test)
    {
        psystem->setIterations(iterations);
        psystem->setDamping(damping);
        psystem->setGravity(-gravity);
        psystem->setCollideNSpring(collideNSpring);
		psystem->setCollideTSpring(collideTSpring);
        psystem->setCollideNDamping(collideNDamping);
        psystem->setCollideTDamping(collideTDamping);
		psystem->setCollideShear(collideShear);
		psystem->setColoumbStaticFric(collideColoumb);
        psystem->setCollideAttraction(collideAttraction);
		psystem->setBoundarySpring(boundarySpring);
		psystem->setBoundaryDamping(boundaryDamping);
		psystem->setBoundaryShear(boundaryShear);
		psystem->setBoundaryColoumb(boundaryColoumb);
		psystem->setCollideRollingFriction(collideRollingFriction);
		psystem->setBoundaryRollingFriction(boundaryRollingFriction);
    	return;
    }
    else
    {

        // create a new parameter list
        params = new ParamListGL("misc");
        params->AddParam(new Param<float>("time step", timestep, 0.00000f, 1.0f, 0.01f, &timestep));
        params->AddParam(new Param<float>("damping"  , damping , 0.0f, 1.0f, 0.01f, &damping));
        params->AddParam(new Param<float>("gravity"  , gravity , 0.0f, 20.0f, 0.1f, &gravity));

        params->AddParam(new Param<float>("collide normal spring" , collideNSpring , 0.0f, 10.0f, 0.1f, &collideNSpring));
		params->AddParam(new Param<float>("collide tangential spring", collideTSpring, 0.0f, 10.0f, 0.1f, &collideTSpring));
        params->AddParam(new Param<float>("collide damping", collideNDamping, 0.0f, 1.0f, 0.01f, &collideNDamping));
		params->AddParam(new Param<float>("collide shear", collideShear, 0.0f, 1.0f, 0.01f, &collideShear));
		params->AddParam(new Param<float>("Coloumb sliding fric", collideColoumb, 0.0f, 1.0f, 0.01f, &collideColoumb));
        params->AddParam(new Param<float>("collide attract", collideAttraction, 0.0f, 0.1f, 0.001f, &collideAttraction));

		params->AddParam(new Param<float>("boundary spring", boundarySpring , 0.0f,10.0f, 0.1f, &boundarySpring));
		params->AddParam(new Param<float>("boundary damping", boundaryDamping , 0.0f, 1.0f, 0.1f, &boundaryDamping));
		params->AddParam(new Param<float>("boundary shear", boundaryShear, 0.0f, 1.0f, 0.01f, &boundaryShear));
    }
}

void mainMenu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Reset block [1]", '1');
    glutAddMenuEntry("Reset random [2]", '2');
	glutAddMenuEntry("Add particles [3]", '3');
	glutAddMenuEntry("Toggle feed [4]", '4');
    glutAddMenuEntry("View mode [v]", 'v');
	glutAddMenuEntry("Record video [k]", 'k');
    glutAddMenuEntry("Toggle point rendering [p]", 'p');
    glutAddMenuEntry("Toggle simulation [ ]", ' ');
    glutAddMenuEntry("Step simulation [ret]", 13);
    glutAddMenuEntry("Toggle sliders [h]", 'h');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    printf("%s Starting...\n\n", sSDKsample);

    numParticles = NUM_PARTICLES;
    uint gridDim = GRID_SIZE;
    numIterations = 0;
    bool nodisplay = false;

	//putenv("NSIGHT_CUDA_DEBUGGER=1");

	g_video_memory_ptr = new unsigned char[(int)width * (int)height * 3 * (int)g_video_fps * g_video_seconds_total];
	g_video_memory_start = g_video_memory_ptr;
	
	cudaDeviceReset();

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "n"))
        {
            numParticles = getCmdLineArgumentInt(argc, (const char **)argv, "n");
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "grid"))
        {
            gridDim = getCmdLineArgumentInt(argc, (const char **) argv, "grid");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", &g_refFile);
            fpsLimit = frameCheckNumber;
            numIterations = 1;
        }

        if (checkCmdLineFlag(argc, (const char **) argv, "nodisplay"))
        {
            nodisplay = true;
            displayEnabled = false;
        }

		if (checkCmdLineFlag(argc, (const char**)argv, "restart"))
		{
			getCmdLineArgumentString(argc, (const char **)argv, "restart", &g_restartFile);
			std::string name = g_restartFile;
			std::cout << "Opening file contents: " << name << std::endl;
			restartFile.open(g_restartFile, std::fstream::binary);
			if(restartFile.is_open())
			{
				std::cout << "Restart file opened successfully.\n" << std::endl;
				do
				{
					if(restartFile.good())
					{
						secondToLastPos = (long long int) restartFile.tellg();
					}
					restartFile.read((char *)numberParticles, sizeof(float));
					restartFile.seekg(*numberParticles*sizeof(float)*9.0f, restartFile.cur);

					lastPos = restartFile.tellg();
					restartFile.read((char *)nextNumParticles, sizeof(float));
					restartFile.seekg(*nextNumParticles*sizeof(float)*9.0f, restartFile.cur);
					if(restartFile.good())
					{
						restartFile.seekg(lastPos);
					}
					else
					{
						break;
					}
					number++;
				} while(restartFile.good());

				printf("Last good frame was frame number %i, at %f seconds, with %f particles.\n\n", number, recordStep*number*timestep, (float)*numberParticles);
				std::cout << "Copying frame contents...";
				restartFile.clear(restartFile.goodbit);
				frame = secondToLastPos-(long long int)(*numberParticles*sizeof(float)*9.0f+sizeof(float));

				restartFile.seekg(0);
				printf("Frame bytes: %lld\n",frame);

				numParticles = (int)*numberParticles;

				pos = new float[numParticles*sizeof(float)*4];
				vel = new float[numParticles*sizeof(float)*4];
				omegvel = new float[numParticles*sizeof(float)*4];
				mass = new float[numParticles*sizeof(float)];
				col = new float[numParticles*sizeof(float)*4];
				memset(pos, 0, numParticles*sizeof(float)*4);
				memset(vel, 0, numParticles*sizeof(float)*4);
				memset(omegvel, 0, numParticles*sizeof(float)*4);
				memset(mass, 0, numParticles*sizeof(float));
				memset(col, 0, numParticles*sizeof(float)*4);

				float *throwaway = new float[1];
				restartFile.read((char *)throwaway, sizeof(float));

				// Get the data from the second to last frame
				for (int u = 0; u < numParticles; u++)
				{
					restartFile.read((char *)r, sizeof(float));
					restartFile.read((char *)i, sizeof(float));
					restartFile.read((char *)type, sizeof(float));
					restartFile.read((char *)x, sizeof(float));
					restartFile.read((char *)y, sizeof(float));
					restartFile.read((char *)z, sizeof(float));
					restartFile.read((char *)vx, sizeof(float));
					restartFile.read((char *)vy, sizeof(float));
					restartFile.read((char *)vz, sizeof(float));

					pos[u*4+0] = *x;
					pos[u*4+1] = *y;
					pos[u*4+2] = *z;
					pos[u*4+3] = *r;
					vel[u*4+0] = *vx;
					vel[u*4+1] = *vy;
					vel[u*4+2] = *vz;
					vel[u*4+3] = *type;
					omegvel[u*4+0] = 0.0f;
					omegvel[u*4+1] = 0.0f;
					omegvel[u*4+2] = 0.0f;
					omegvel[u*4+3] = (float)u;

					if (*type == 1.0f)
					{
						col[u*4+0] = 1.0f;
						col[u*4+1] = 0.2f;
						col[u*4+2] = 0.2f;
						col[u*4+3] = 1.0f;
					}
					else
					{
						col[u*4+0] = 0.2f;
						col[u*4+1] = 0.2f;
						col[u*4+2] = 1.0f;
						col[u*4+3] = 1.0f;
					}

					mass[u] = 4.0f / 3.0f * 3.141592654f * powf(*r, 3) * 2500.0f;

				}

				lastPos = (long long int) restartFile.tellg();
				printf("Last position: %lld\n",lastPos);
				printf("Second to last position: %lld\n",secondToLastPos);
				restartFile.close();
				std::cout << "Frame contents successfully copied to memory.\n" << std::endl;
				//printf("Test particle 5000: x = %f, y = %f, z = %f, and r = %f\n", pos[5000*4+0], pos[5000*4+1], pos[5000*4+2], pos[5000*4+3]);
			}
			else
			{
				std::cout << "Failed to open restart file." << std::endl;
			}
		}
    }

    gridSize.x = 8;
	gridSize.y = 64;
	gridSize.z = 64;
    printf("grid: %d x %d x %d = %d cells\n", gridSize.x, gridSize.y, gridSize.z, gridSize.x*gridSize.y*gridSize.z);
    printf("particles: %d\n", numParticles);

    bool benchmark = checkCmdLineFlag(argc, (const char **) argv, "benchmark") != 0;

    if (checkCmdLineFlag(argc, (const char **) argv, "i"))
    {
        numIterations = getCmdLineArgumentInt(argc, (const char **) argv, "i");
    }
    if (g_refFile || benchmark)
    {
        cudaInit(argc, argv);
    }
    if (nodisplay)
    {
    	cudaInit(argc, argv);
    }
    else
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            printf("[%s]\n", argv[0]);
            printf("   Does not explicitly support -device=n in OpenGL mode\n");
            printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
            printf(" > %s -device=n -file=<*.bin>\n", argv[0]);
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }

        initGL(&argc, argv);
        cudaGLInit(argc, argv);
    }

    initParticleSystem(numParticles, gridSize, !nodisplay);
    initParams(nodisplay);

//    if (!g_refFile)
//    {
//        initMenus();
//    }

    if (benchmark || g_refFile)
    {
        if (numIterations <= 0)
        {
            numIterations = 3000;
        }

        runBenchmark(numIterations, argv[0]);
    }
    if (nodisplay)
    {
    	runTest(numIterations);
    }
    else
    {
        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutSpaceballMotionFunc(spacemotion);
        glutSpaceballRotateFunc(spacerotate);
        glutKeyboardFunc(key);
        glutSpecialFunc(special);
        glutIdleFunc(idle);

        atexit(cleanup);

        glutMainLoop();
    }

    if (psystem)
    {
        delete psystem;
    }

	delete [] pos;
	delete [] vel;
	delete [] col;
	delete [] mass;
	delete [] omegvel;

    cudaDeviceReset();
    exit(g_TotalErrors > 0 ? EXIT_FAILURE : EXIT_SUCCESS);
}

