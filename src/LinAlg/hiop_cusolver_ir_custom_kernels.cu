//written by KS some time in 2019
#define maxk 80
#define Tv5 1024
//computes V^T[u1 u2] where v is n x k and u1 and u2 are nx1
__global__ void MassIPTwoVec(const double* __restrict__ u1, 
                             const double* __restrict__ u2, 
                             const double* __restrict__ 
                             v, double* result,
                             const int k, 
                             const int N)
{
  int t = threadIdx.x;
  int bsize = blockDim.x;

  // assume T threads per thread block (and k reductions to be performed)
  volatile __shared__ double s_tmp1[Tv5];

  volatile __shared__ double s_tmp2[Tv5];
  // map between thread index space and the problem index space
  int j = blockIdx.x;
  s_tmp1[t] = 0.0f;
  s_tmp2[t] = 0.0f;
  int nn = t;
  double can1, can2, cbn;

  while(nn < N) {
    can1 = u1[nn];
    can2 = u2[nn];

    cbn = v[N * j + nn];
    s_tmp1[t] += can1 * cbn;
    s_tmp2[t] += can2 * cbn;

    nn += bsize;
  }

  __syncthreads();

  if(Tv5 >= 1024) {
    if(t < 512) {
      s_tmp1[t] += s_tmp1[t + 512];
      s_tmp2[t] += s_tmp2[t + 512];
    }
    __syncthreads();
  }
  if(Tv5 >= 512) {
    if(t < 256) {
      s_tmp1[t] += s_tmp1[t + 256];
      s_tmp2[t] += s_tmp2[t + 256];
    }
    __syncthreads();
  }
  {
    if(t < 128) {
      s_tmp1[t] += s_tmp1[t + 128];
      s_tmp2[t] += s_tmp2[t + 128];
    }
    __syncthreads();
  }
  {
    if(t < 64) {
      s_tmp1[t] += s_tmp1[t + 64];
      s_tmp2[t] += s_tmp2[t + 64];
    }
    __syncthreads();
  }

  if(t < 32) {
    s_tmp1[t] += s_tmp1[t + 32];
    s_tmp2[t] += s_tmp2[t + 32];

    s_tmp1[t] += s_tmp1[t + 16];
    s_tmp2[t] += s_tmp2[t + 16];

    s_tmp1[t] += s_tmp1[t + 8];
    s_tmp2[t] += s_tmp2[t + 8];

    s_tmp1[t] += s_tmp1[t + 4];
    s_tmp2[t] += s_tmp2[t + 4];

    s_tmp1[t] += s_tmp1[t + 2];
    s_tmp2[t] += s_tmp2[t + 2];

    s_tmp1[t] += s_tmp1[t + 1];
    s_tmp2[t] += s_tmp2[t + 1];
  }
  if(t == 0) {
    result[blockIdx.x] = s_tmp1[0];
    result[blockIdx.x + k] = s_tmp2[0];
  }
}

// same as previous one but also scales the result by scaleFactors

__global__ void MassIPV7part1withScaling(const double* __restrict__ u,
                                         const double* __restrict__ v, 
                                         const double* scaleFactors,
                                         double* result, 
                                         const int k, 
                                         const int N)
{

  //  int b = blockIdx.x;
  int t = threadIdx.x;
  int bsize = blockDim.x;

  // assume T threads per thread block (and k reductions to be performed)
  volatile __shared__ double s_tmp[Tv5];
  double s;
  // map between thread index space and the problem index space
  int j = blockIdx.x;
  s_tmp[t] = 0.0f;
  int nn = t;

  while(nn < N) {
    double can = u[nn];

    double cbn = v[N * j + nn];
    s_tmp[t] += can * cbn;
    if((nn + bsize) < N) {
      can = u[nn + bsize];
      cbn = v[N * j + (nn + bsize)];
      s_tmp[t] += can * cbn;
    }

    nn += 2 * bsize;
  }

  __syncthreads();
  if(Tv5 >= 1024) {
    if(t < 512) {
      s_tmp[t] += s_tmp[t + 512];
    }
    __syncthreads();
  }
  if(Tv5 >= 512) {
    if(t < 256) {
      s_tmp[t] += s_tmp[t + 256];
    }
    __syncthreads();
  }
  {
    if(t < 128) {
      s_tmp[t] += s_tmp[t + 128];
    }
    __syncthreads();
  }
  {
    if(t < 64) {
      s_tmp[t] += s_tmp[t + 64];
    }
    __syncthreads();
  }

  if(t < 32) {
    s_tmp[t] += s_tmp[t + 32];
    s_tmp[t] += s_tmp[t + 16];

    s_tmp[t] += s_tmp[t + 8];
    s_tmp[t] += s_tmp[t + 4];
    s_tmp[t] += s_tmp[t + 2];
    s_tmp[t] += s_tmp[t + 1];
  }
  if(t == 0) {
    s = scaleFactors[blockIdx.x];
    result[blockIdx.x] = s * s_tmp[0];
  }
}

//mass AXPY i.e y = y - x*alpha where alpha is [k x 1], needed in 1 and 2 synch GMRES

__global__ void massAxpy3(int N,
                          int k,
                          const double* x_data,
                          double* y_data,
                          const double* alpha) {

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int t = threadIdx.x;
  __shared__ double s_alpha[maxk];
  if(t < k) {
    s_alpha[t] = alpha[t];
  }
  __syncthreads();

  if(i < N) {
    double temp = 0.0f;
    for(int j = 0; j < k; ++j) {
      temp += x_data[j * N + i] * s_alpha[j];
    }
    y_data[i] -= temp;
  }
}

void massInnerProductTwoVectors(int n, 
                                int i, 
                                double* vec1, 
                                double* vec2, 
                                double* mvec, 
                                double* result)
{
  MassIPTwoVec<<<i + 1, 1024> > >(vec1, vec2, mvec, result, i + 1, n);
}
void massAxpy(int n, int i, double* x, double* y, double* alpha)
{
  massAxpy3<<<(n + 384 - 1) / 384, 384> > >(n, i + 1, x, y, alpha);
}

