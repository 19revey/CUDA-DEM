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

#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(
                               attribute float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    gl_FrontColor = gl_Color;
}
                           );

// vertex shader
const char *vertexShader1 = STRINGIFY(
                               attribute float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	if (gl_Color.x == 1.0 && gl_Color.y == 0.2 && gl_Color.z == 0.2 && gl_Color.a == 1.0)
	{
		gl_FrontColor = gl_Color;
	}
	else
	{
		gl_FrontColor = gl_Color * vec4(0.0, 0.0, 0.0, 0.0);
	}
}
                           );
						   			   
// vertex shader
const char *vertexShader2 = STRINGIFY(
                               attribute float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	if (gl_Color.x == 0.2 && gl_Color.y == 0.2 && gl_Color.z == 1.0 && gl_Color.a == 1.0)
	{
		gl_FrontColor = gl_Color;
	}
	else
	{
		gl_FrontColor = gl_Color * vec4(0.0, 0.0, 0.0, 0.0);
	}
}
                           );

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(
									attribute float pointRadius;
                                    void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

	//GLfloat whiteSpecularLight[] = {1.0, 1.0, 1.0, 0.0};

	//glMaterialfv(GL_FRONT, GL_SPECULAR, const (GLfloat*) whiteSpecularLight);
    gl_FragColor = gl_Color * diffuse * pointRadius / pointRadius;
}
                                );
// pixel shader to render only the red (small) spheres
const char *spherePixelShader1 = STRINGIFY(
									attribute float pointRadius;
									void main()
{
	const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

	//GLfloat whiteSpecularLight[] = {1.0, 1.0, 1.0, 0.0};
	//glMaterialfv(GL_FRONT, GL_SPECULAR, const (GLfloat*) whiteSpecularLight);

	if (gl_Color.x == 1.0 && gl_Color.y == 0.2 && gl_Color.z == 0.2 && gl_Color.a == 1.0)
	{
		gl_FragColor = gl_Color * diffuse * pointRadius / pointRadius;
	}
	else
	{
		gl_FragColor = gl_Color * vec4(1.0, 1.0, 1.0, 0.0);
	}

}
								);
// pixel shader to render only the blue (large) spheres
const char *spherePixelShader2 = STRINGIFY(
									attribute float pointRadius;
									void main()
{
	const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

	//GLfloat whiteSpecularLight[] = {1.0, 1.0, 1.0, 0.0};
	//glMaterialfv(GL_FRONT, GL_SPECULAR, const (GLfloat*) whiteSpecularLight);

	if (gl_Color.x == 0.2 && gl_Color.y == 0.2 && gl_Color.z == 1.0 && gl_Color.a == 1.0)
	{
		gl_FragColor = gl_Color * diffuse * pointRadius / pointRadius;
	}
	else
	{
		gl_FragColor = gl_Color * vec4(1.0, 1.0, 1.0, 0.0);
	}

}
								);
// pixel shader for rendering points as shaded spheres colored according to their velocity
const char *spherePixelShader3 = STRINGIFY(
									attribute float pointRadius;
                                    void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

	float colz = abs(gl_Color.z);

	float colzn = colz * cos(0.34906) / 0.4;

    gl_FragColor = vec4(colzn, colzn, 1.0, 1.0) * diffuse * pointRadius / pointRadius;
}
                                );