#pragma once

#include <cmath>
#include <vector>

typedef unsigned char byte;

class ByteImage
{
public:
	ByteImage(unsigned char* imageData, int width, int height, int channels);
	ByteImage(int width, int height, int channels);
	ByteImage(const ByteImage& rhs);
	~ByteImage();

	// return -1 if out of boarder
	inline int Pixel(int x, int y, int channel)
	{
		if (x < 0 || x >= m_width) return -1;
		if (y < 0 || y >= m_height) return -1;
		if (channel < 0 || channel >= m_channels) return -1;

		return m_imageData[(y * m_width + x) * m_channels + channel];
	}

	inline int SetPixel(int x, int y, int channel, int value)
	{
		if (x < 0 || x >= m_width) return -1;
		if (y < 0 || y >= m_height) return -1;
		if (channel < 0 || channel >= m_channels) return -1;

		m_imageData[(y * m_width + x) * m_channels + channel] = value & 0xff;
		return value;
	}

	std::vector<float> Avarage();

	static bool PoissonFilter(ByteImage& dst, ByteImage& src);

	unsigned char* Data() { return m_imageData; }

private:
	friend class FloatImage;
	unsigned char* m_imageData;
	int m_width;
	int m_height;
	int m_channels;
	bool m_needFreeData;
};

class FloatImage
{
public:
	FloatImage(int width, int height, int channels, float initValues=0);
	FloatImage(const FloatImage& rhs);
	~FloatImage();

	// return NAN if out of boarder
	inline float Pixel(int x, int y, int channel)
	{
		if (x < 0 || x >= m_width) return NAN;
		if (y < 0 || y >= m_height) return NAN;
		if (channel < 0 || channel >= m_channels) return NAN;

		return m_imageData[(y * m_width + x) * m_channels + channel];
	}

	inline float SetPixel(int x, int y, int channel, float value)
	{
		if (x < 0 || x >= m_width) return NAN;
		if (y < 0 || y >= m_height) return NAN;
		if (channel < 0 || channel >= m_channels) return NAN;

		m_imageData[(y * m_width + x) * m_channels + channel] = value;
		return value;
	}

	unsigned char *ToBytes(float scale=1, std::vector<float> *offset = nullptr, bool abs=false);

	static bool PoissonFilter(FloatImage& dst, ByteImage& src);
	FloatImage JacobiCpu(int numIters=100);
	FloatImage JacobiCuda(int numIters = 100);
	FloatImage SorCuda(int numIters = 100);
	FloatImage ConjugateGradientCuda(int numIters = 100);

	float *Data() { return m_imageData; }

private:
	float *m_imageData;
	int m_width;
	int m_height;
	int m_channels;
	bool m_needFreeData;
};