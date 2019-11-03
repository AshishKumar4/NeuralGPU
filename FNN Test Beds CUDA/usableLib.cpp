#include "usableLib.h"
#include "stdlib.h"
#include "stdio.h"
#include "stdint.h"
#include "string.h"
#include "iostream"
#include "math.h"
#include "cmath"

float* oneHotEncode(uint8_t value, int sz)
{
	float* val = new float[sz];
	memset(val, 0, sizeof(float)*sz);
	val[value] = 1;
	return val;
}

uint8_t oneHotDecode(float* val, int sz)
{
	float a = val[0];
	int b = 0;
	for(int i = 0; i < sz; i++)
	{
		if(val[i] > a)
		{
			a = val[i];
			b = i;
		}
	}
	return (uint8_t)b;
}

float calcMean(uint8_t* arr, int n)
{
	int s = 0;
	for(int i = 0; i < n; i++)
	{
		s += arr[i];
	}
	return float(s)/float(n);
}

float standardDeviation(uint8_t* arr, int n)
{
	float mean = calcMean(arr, n);
	float s = 0;
	for(int i = 0; i < n; i++)
	{
		s += powf(float(arr[i]) - mean, 2); 
	}
	s /= n;
	return sqrt(s);
}

float* normalizeArray(uint8_t* arr, int n)
{
	float mean = calcMean(arr, n);
	float stdv = standardDeviation(arr, n);
	// printf("\n==>mean = %f, stdv = %f", mean, stdv);
	float* result = new float[n];
	for(int i = 0; i < n; i++)
	{
		result[i] = (float(arr[i]) / 255.);//- mean) / stdv;
	}
	return result;
}

char tbuf[32];
char bchars[] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};

void itoa(unsigned i,char* buf, unsigned base)
{
    int pos = 0;
    int opos = 0;
    int top = 0;

    if (i == 0 || base > 16) {
        buf[0] = '0';
        buf[1] = '\0';
        return;
    }

    while (i != 0) {
        tbuf[pos] = bchars[i % base];
        pos++;
        i /= base;
    }
    top=pos--;
    for (opos=0; opos<top; pos--,opos++)
        {
        buf[opos] = tbuf[pos];
    }
    buf[opos] = 0;
}

float* NN_FeedEncoder(int n, int base, int sz, float lower_bound, float upper_bound)
{
	char* buf = new char[sz + 1];

	itoa(n, buf, base);
	//printf("\n{%d; %s}", n, buf);
	float* bb = new float[sz];
	int i = strlen(buf), _i = sz - i;
	//printf("\t[%d %d]", i, _i);
	for (int j = 0; j < _i; j++)	// String from itoa is null terminated
	{
		bb[j] = lower_bound;
	}
	for (int j = 0; j < i; j++)
	{
		char tt[2];
		strncpy(tt, &buf[j], 1);
		bb[_i + j] = ((atof(tt) / (base - 1))*(upper_bound - lower_bound)) - ((upper_bound - lower_bound)/2);
		//cout << "-" << bb[_i + j];
	}
	//cout << endl;

	delete buf;
	return bb;
}

int NN_FeedDecoder(float* val, int base, int sz, float lower_bound, float upper_bound)	// Currently only binary
{
	int tt = 0;
	for (int i = 0; i < sz; i++)
	{
		if (abs(upper_bound - val[i]) < float(upper_bound - lower_bound)/float(base))
		{
			tt += pow(2, sz - i - 1) * (base - 1);
		}
		else if (abs(lower_bound - val[i]) < float(upper_bound - lower_bound) / float(base))
		{
			//tt += pow(2, sz - i - 1) * (base - 2);
		}
	}
	return tt;
}
