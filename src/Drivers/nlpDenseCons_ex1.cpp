#include "nlpDenseCons_ex1.hpp"

#include <cmath>
#include <cstdio>
#include <cassert>

using namespace hiop;

Ex1Meshing1D::Ex1Meshing1D(double a, double b, 
			   long long glob_n, double r, 
			   MPI_Comm comm_)
{
  _a=a; _b=b; _r=r;
  comm=comm_;
  comm_size=1; my_rank=0; 
#ifdef HIOP_USE_MPI
  int ierr = MPI_Comm_size(comm, &comm_size); assert(MPI_SUCCESS==ierr);
  ierr = MPI_Comm_rank(comm, &my_rank); assert(MPI_SUCCESS==ierr);
#endif
  // set up vector distribution for primal variables - easier to store it as a member in this simple example
  col_partition = new long long[comm_size+1];
  long long quotient=glob_n/comm_size, remainder=glob_n-comm_size*quotient;
  
  int i=0; col_partition[i]=0; i++;
  while(i<=remainder) { col_partition[i] = col_partition[i-1]+quotient+1; i++; }
  while(i<=comm_size) { col_partition[i] = col_partition[i-1]+quotient;   i++; }

  _mass = LinearAlgebraFactory::createVector(glob_n, col_partition, comm);

  //if(my_rank==0) printf("reminder=%d quotient=%d\n", remainder, quotient);
  //printf("left=%d right=%d\n", col_partition[my_rank], col_partition[my_rank+1]);

  //compute the mass
  double m1=2*_r / ((1+_r)*glob_n);
  double h =2*(1-_r) / (1+_r) / (glob_n-1) / glob_n;

  long long glob_n_start=col_partition[my_rank], glob_n_end=col_partition[my_rank+1]-1;

  double* mass = _mass->local_data(); //local slice
  double rescale = _b-_a;
  for(long long k=glob_n_start; k<=glob_n_end; k++) {
    mass[k-glob_n_start] = (m1 + (k-glob_n_start)*h) * rescale;
    //printf(" proc %d k=%d  mass[k]=%g\n", my_rank, k, mass[k-glob_n_start]);
  }

  //_mass->print(stdout, NULL);
  //fflush(stdout);
}
Ex1Meshing1D::~Ex1Meshing1D()
{
  delete[] col_partition;
  delete _mass;
}

bool Ex1Meshing1D::get_vecdistrib_info(long long global_n, long long* cols)
{
  for(int i=0; i<=comm_size; i++) cols[i] = col_partition[i];
  return true;
}
void Ex1Meshing1D::applyM(DiscretizedFunction& f)
{
  f.componentMult(*this->_mass);
}

//converts the local indexes to global indexes
long long Ex1Meshing1D::getGlobalIndex(long long i_local) const
{
  assert(0<=i_local); 
  assert(i_local < col_partition[my_rank+1]-col_partition[my_rank]);

  return i_local+col_partition[my_rank];
}

long long Ex1Meshing1D::getLocalIndex(long long i_global) const
{
  assert(i_global >= col_partition[my_rank]);
  assert(i_global <  col_partition[my_rank+1]);
  return i_global-col_partition[my_rank];
}

//for a function c(t), for given global index in the discretization 
// returns the corresponding continuous argument 't', which is in this 
// case the middle of the discretization interval.
double Ex1Meshing1D::getFunctionArgument(long long i_global) const
{
  assert(i_global >= col_partition[my_rank]);
  assert(i_global <  col_partition[my_rank+1]);

  const long & k = i_global;

  long long glob_n = size();
  double m1=2*_r / ((1+_r)*glob_n);
  double h =2*(1-_r) / (1+_r) / (glob_n-1) / glob_n;

  //t is the middle of [k*m1 + k(k-1)/2*h, (k+1)m1+ (k+1)k/2*h]
  double t = 0.5*( (2*k+1)*m1 + k*k*h);
  return t;
}



/* DiscretizedFunction implementation */

DiscretizedFunction::DiscretizedFunction(Ex1Meshing1D* meshing)
  : hiopVectorPar(meshing->size(), meshing->get_col_partition(), meshing->get_comm())
{
  _mesh = meshing;
}

// u'*v = u'*M*v, where u is 'this'
double DiscretizedFunction::dotProductWith( const DiscretizedFunction& v_ ) const
{
  assert(v_._mesh->matches(this->_mesh));
  double* M=_mesh->_mass->local_data();
  double* u= this->data_;
  double* v= v_.data_;
  
  double dot=0.;
  for(int i=0; i<get_local_size(); i++)
    dot += u[i]*M[i]*v[i];
 
 #ifdef HIOP_USE_MPI
  double dotprodG;
  int ierr = MPI_Allreduce(&dot, &dotprodG, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  dot=dotprodG;
#endif
  return dot;
}

// computes integral of 'this', that is sum (this[elem]*m[elem])
double DiscretizedFunction::integral() const
{
  //the base dotProductWith method would do it
  return hiopVectorPar::dotProductWith(*_mesh->_mass);
}

// norm(u) as sum(M[elem]*u[elem]^2)
double DiscretizedFunction::twonorm() const 
{
  double* M=_mesh->_mass->local_data();
  double* u= this->data_;

  double nrm_square=0.;
  for(int i=0; i<get_local_size(); i++)
    nrm_square += u[i]*u[i]*M[i];

#ifdef HIOP_USE_MPI
  double nrm_squareG;
  int ierr = MPI_Allreduce(&nrm_square, &nrm_squareG, 1, MPI_DOUBLE, MPI_SUM, comm_); assert(MPI_SUCCESS==ierr);
  nrm_square=nrm_squareG;
#endif  
  return sqrt(nrm_square);
}


//converts the local indexes to global indexes
long long DiscretizedFunction::getGlobalIndex(long long i_local) const
{
  return _mesh->getGlobalIndex(i_local);
}

//for a function c(t), for given global index in the discretization 
// returns the corresponding continuous argument 't', which is in this 
// case the middle of the discretization interval.
double DiscretizedFunction::getFunctionArgument(long long i_global) const
{
  return _mesh->getFunctionArgument(i_global);
}

//set the function value for a given global index
void DiscretizedFunction::setFunctionValue(long long i_global, const double& value)
{
  long long i_local=_mesh->getLocalIndex(i_global);
  this->data_[i_local]=value;
}



/* Ex1Interface class implementation */

/*set c to  
 *    c(t) = 1-10*t, for 0<=t<=1/10,
 *           0,      for 1/10<=t<=1.
 */
void Ex1Interface::set_c()
{
  for(int i_local=0; i_local<n_local; i_local++) {
    //this will be based on 'my_rank', thus, different ranks get the appropriate global indexes
    long long n_global = c->getGlobalIndex(i_local); 
    double t = c->getFunctionArgument(n_global);
    //if(t<=0.1) c->setFunctionValue(n_global, 1-10.*t);
    double cval;
    if(t<=0.1) cval = -1.+10.*t;
    else       cval = 0.;

    c->setFunctionValue(n_global, cval);
    //printf("index %d  t=%g value %g\n", n_global, t, cval);
  } 
}
