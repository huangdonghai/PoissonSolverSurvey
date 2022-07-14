// PoissonSolverSurvey.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
#include "ImageHelper.h"

int main()
{
    const char* imagePath = "yacht.jpg";
    const char* imageOut = "yacht_poisson.jpg";
	const char* imageJacobi = "yacht_jacobi.jpg";

    int width, height, n;

    unsigned char* imageData = stbi_load(imagePath, &width, &height, &n, 0);
	if (imageData == 0)
		return EXIT_FAILURE;

	ByteImage src(imageData, width, height, n);

	auto avarage = src.Avarage();

	FloatImage dst(width, height, n);

	FloatImage::PoissonFilter(dst, src);

	auto poissonData = dst.ToBytes(1, nullptr, true);

	stbi_write_jpg(imageOut, width, height, n, poissonData, 80);

	auto jacobi = dst.JacobiCuda(200000);

	auto jacobiData = jacobi.ToBytes(1, &avarage, false);

	stbi_write_jpg(imageJacobi, width, height, n, jacobiData, 80);

	std::cout << "Hello World!\n";
}

