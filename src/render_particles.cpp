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

#include <GL/glew.h>
#include <string>

#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "render_particles.h"
#include "shaders.h"

#ifndef M_PI
#define M_PI    3.1415926535897932384626433832795
#endif

using namespace std;

ParticleRenderer::ParticleRenderer()
    : m_pos(0),
      m_numParticles(0),
      m_pointSize(1.0f),
      m_particleRadius(0.125f * 0.5f),
      m_program(0),
      m_vbo(0),
      m_colorVBO(0),
	  m_velocityVBO(0)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    m_pos = 0;
}

void ParticleRenderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setVertexBuffer(unsigned int vbo, int numParticles)
{
    m_vbo = vbo;
    m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints()
{
	//GLint r;
	//GL_FLOAT;

    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
		//GLchar param;
		//GLint r;

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
		//r = glGetAttribLocation(m_vbo, "pointRadius");
		//printf("%i",r);

		glVertexPointer(3, GL_FLOAT, 16, (void*) 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (void*) 12);
		glEnableVertexAttribArray(1);

		//glGetActiveAttrib(m_program, 1, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, NULL, NULL, NULL, &param);
		//printf("%s", param);

		//printf(glGetActiveAttrib(m_program, 1, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, NULL, NULL,, name));

        if (m_colorVBO)
        {
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_colorVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }

        glDrawArrays(GL_POINTS, 0, m_numParticles);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		glDisableVertexAttribArray(1);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
}

void ParticleRenderer::_drawPoints2()
{
    if (!m_vbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;

            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vbo);
		glVertexPointer(3, GL_FLOAT, 16, (void*) 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 16, (void*) 12);
		glEnableVertexAttribArray(1);

        if (m_velocityVBO)
        {
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_velocityVBO);
            glColorPointer(4, GL_FLOAT, 0, 0);
            glEnableClientState(GL_COLOR_ARRAY);
        }

        glDrawArrays(GL_POINTS, 0, m_numParticles);

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
		glDisableVertexAttribArray(1);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */)
{
    switch (mode)
    {
        case PARTICLE_POINTS:
            glColor3f(1, 1, 1);
            glPointSize(m_pointSize);
            _drawPoints();
            break;

        default:
        case PARTICLE_SPHERES:
            glEnable(GL_POINT_SPRITE_ARB);
            glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
            glDepthMask(GL_TRUE);
			//glEnable(GL_DEPTH_CLAMP);
            glEnable(GL_DEPTH_TEST);

            glUseProgram(m_program);
            glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
            //glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);

            glColor3f(1, 1, 1);
            _drawPoints();

            glUseProgram(0);
            glDisable(GL_POINT_SPRITE_ARB);
            break;

		case VELOCITY_MAP:
            glEnable(GL_POINT_SPRITE_ARB);
            glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);

            glUseProgram(m_program3);
            glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));

            glColor3f(1, 1, 1);
            _drawPoints2();

            glUseProgram(0);
            glDisable(GL_POINT_SPRITE_ARB);
            break;

		case PARTICLE_SMALL:
			glEnable(GL_POINT_SPRITE_ARB);
            glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
			glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_BLEND);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);

            glUseProgram(m_program1);
            glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
            //glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);

            glColor3f(1, 1, 1);
            _drawPoints();

            glUseProgram(0);
            glDisable(GL_POINT_SPRITE_ARB);
			glDisable(GL_BLEND);
            break;

		case PARTICLE_LARGE:
			glEnable(GL_POINT_SPRITE_ARB);
            glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
			glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glEnable(GL_BLEND);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);

            glUseProgram(m_program2);
            glUniform1f(glGetUniformLocation(m_program, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI/180.0f));
            //glUniform1f(glGetUniformLocation(m_program, "pointRadius"), m_particleRadius);

            glColor3f(1.0, 1.0, 1.0);
            _drawPoints();

            glUseProgram(0);
            glDisable(GL_POINT_SPRITE_ARB);
			glDisable(GL_BLEND);
            break;
    }
}

GLuint
ParticleRenderer::_compileProgram(const char *vsource, const char *fsource)
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vertexShader, 1, &vsource, 0);
    glShaderSource(fragmentShader, 1, &fsource, 0);

    glCompileShader(vertexShader);
    glCompileShader(fragmentShader);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

	glBindAttribLocation(program, 1, "pointRadius");

    glLinkProgram(program);

    // check if program linked
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success)
    {
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        printf("Failed to link program:\n%s\n", temp);
        glDeleteProgram(program);
        program = 0;
    }

    return program;
}

void ParticleRenderer::_initGL()
{
    m_program = _compileProgram(vertexShader, spherePixelShader);
	m_program1 = _compileProgram(vertexShader1, spherePixelShader1);
	m_program2 = _compileProgram(vertexShader2, spherePixelShader2);
	m_program3 = _compileProgram(vertexShader, spherePixelShader3);

#if !defined(__APPLE__) && !defined(MACOSX)
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}
