float benchmark_data[1000];
static const int N = 1000;

 float compute_mac_flops(float* data){
  float b1  = 11.0f;
  float b2  = 12.0f;
  float b3  = 13.0f;
  float b4  = 14.0f;
  float b5  = 15.0f;
  float b6  = 16.0f;
  float b7  = 17.0f;
  float b8  = 18.0f;
  float b9  = 19.0f;
  float b10 = 110.0f;
  float b11 = 111.0f;
  float b12 = 112.0f;
  float b13 = 113.0f;
  float b14 = 114.0f;
  float b15 = 115.0f;
  float b16 = 116.0f;

  float accum1  = 0;
  float accum2  = 0;
  float accum3  = 0;
  float accum4  = 0;
  float accum5  = 0;
  float accum6  = 0;
  float accum7  = 0;
  float accum8  = 0;
  float accum9  = 0;
  float accum10 = 0;
  float accum11 = 0;
  float accum12 = 0;
  float accum13 = 0;
  float accum14 = 0;
  float accum15 = 0;
  float accum16 = 0;

  for (int i = 0; i < N; i+=16) {
      accum1  += b1  * data[i + 0 ];
      accum2  += b2  * data[i + 1 ];
      accum3  += b3  * data[i + 2 ];
      accum4  += b4  * data[i + 3 ];
      accum5  += b5  * data[i + 4 ];
      accum6  += b6  * data[i + 5 ];
      accum7  += b7  * data[i + 6 ];
      accum8  += b8  * data[i + 7 ];
      accum9  += b9  * data[i + 8 ];
      accum10 += b10 * data[i + 9 ];
      accum11 += b11 * data[i + 10];
      accum12 += b12 * data[i + 11];
      accum13 += b13 * data[i + 12];
      accum14 += b14 * data[i + 13];
      accum15 += b15 * data[i + 14];
      accum16 += b16 * data[i + 15];
  }

  float total = accum1 + accum2 + accum3 + accum4 + accum5 + accum6 + accum7 + accum8 + accum9 + accum10 + accum11 + accum12 + accum13 + accum14 + accum15 + accum16;
  return total;
}
