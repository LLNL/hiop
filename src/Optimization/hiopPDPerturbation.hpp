#ifndef HIOP_PERTURB_PD_LINSSYS
#define HIOP_PERTURB_PD_LINSSYS

#include "hiopNlpFormulation.hpp"
#include "hiopVector.hpp"

namespace hiop
{

class hiopPDPerturbationBase
{
public:
  /** Default constructor 
   * Provides complete initialization, but uses algorithmic parameters from the Ipopt
   * implementation paper. \ref initialize(hiopNlpFormulation*) should be used to initialize
   * this class based on HiOp option file or HiOp user-supplied runtime options.
   */
  hiopPDPerturbationBase()
    : delta_w_min_bar_(1e-20),
      delta_w_max_bar_(1e+40),
      delta_w_0_bar_(1e-4),
      kappa_w_minus_(1./3),
      kappa_w_plus_bar_(100.),
      kappa_w_plus_(8.),
      delta_c_bar_(1e-8),
      kappa_c_(0.25),
      hess_degenerate_(dtNotEstablished),
      jac_degenerate_(dtNotEstablished),
      num_degen_iters_(0),
      num_degen_max_iters_(3),
      deltas_test_type_(dttNoTest),
      mu_(1e-8),
      delta_wx_curr_(nullptr),
      delta_wd_curr_(nullptr),
      delta_cc_curr_(nullptr),
      delta_cd_curr_(nullptr),
      delta_wx_last_(nullptr),
      delta_wd_last_(nullptr),
      delta_cc_last_(nullptr),
      delta_cd_last_(nullptr),
      nlp_(nullptr)
  {
  }

  virtual ~hiopPDPerturbationBase()
  {
    delete delta_wx_curr_;
    delete delta_wd_curr_;
    delete delta_cc_curr_;
    delete delta_cd_curr_;
    delete delta_wx_last_;
    delete delta_wd_last_;
    delete delta_cc_last_;
    delete delta_cd_last_;
  }

  /** Initializes and reinitializes object based on the 'options' parameters of the
   * 'nlp_' object.
   * Returns 'false' if something goes wrong, otherwise 'true'
   */
  virtual bool initialize(hiopNlpFormulation* nlp);

  /** Set log-barrier mu. */
  virtual inline void set_mu(const double& mu)
  {
    mu_ = mu;
  }

  /** Called when a new linear system is attempted to be factorized 
   */
  virtual bool compute_initial_deltas(hiopVector& delta_wx, 
                                      hiopVector& delta_wd,
                                      hiopVector& delta_cc,
                                      hiopVector& delta_cd) = 0;

  /** Method for correcting inertia */
  virtual bool compute_perturb_wrong_inertia(hiopVector& delta_wx, 
                                             hiopVector& delta_wd,
                                             hiopVector& delta_cc,
                                             hiopVector& delta_cd) = 0;

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  virtual bool compute_perturb_singularity(hiopVector& delta_wx, 
                                           hiopVector& delta_wd,
                                           hiopVector& delta_cc,
                                           hiopVector& delta_cd) = 0;

  inline bool get_curr_perturbations(hiopVector& delta_wx, 
                                     hiopVector& delta_wd,
                                     hiopVector& delta_cc,
                                     hiopVector& delta_cd)
  {
    delta_wx.copyFrom(*delta_wx_curr_);
    delta_wd.copyFrom(*delta_wd_curr_);
    delta_cc.copyFrom(*delta_cc_curr_);
    delta_cd.copyFrom(*delta_cd_curr_);
    return true;
  }

protected:
  /** Current and last perturbations, primal is split in x and d, dual in c and d. */
  hiopVector* delta_wx_curr_;
  hiopVector* delta_wd_curr_;
  hiopVector* delta_cc_curr_;
  hiopVector* delta_cd_curr_;

  hiopVector* delta_wx_last_;
  hiopVector* delta_wd_last_;
  hiopVector* delta_cc_last_;
  hiopVector* delta_cd_last_;

  /** Algorithmic parameters */

  /** Smallest possible perturbation for Hessian (for primal 'x' and 's' variables). */
  double delta_w_min_bar_;
  /** Maximal perturbation for for Hessian (for primal 'x' and 's' variables). */
  double delta_w_max_bar_;
  /** First trial value for delta_w perturbation. */
  double delta_w_0_bar_;
  /** Decrease factor for delta_w. */
  double kappa_w_minus_;
  /** Increase factor for delta_w for first required perturbation. */
  double kappa_w_plus_bar_;
  /** Increase factor for delta_w for later perturbations. */
  double kappa_w_plus_;
  
  /** Factor for regularization for potentially rank-deficient Jacobian. */
  double delta_c_bar_;
  /** Exponent of mu when computing regularization for Jacobian. */
  double kappa_c_;
  
  /** Degeneracy is handled as in Ipopt*/
  
  /** Type for degeneracy flags */
  enum DegeneracyType
  {
    dtNotEstablished,
    dtNotDegenerate,
    dtDegenerate
  };
  /** Structural degeneracy of the Hessian */
  DegeneracyType hess_degenerate_;

  /** Structural degeneracy of the Jacobian */
  DegeneracyType jac_degenerate_;

  /* Counter for inertia corrections at which the matrix is suspected to be degenerate. */
  int num_degen_iters_;
  /* Max number of iters after which to conclude matrix is degenerate. */
  const int num_degen_max_iters_;
  
  /** Status of current trial configuration */
  enum DeltasTestType
  {
    dttNoTest,
    dttDeltac0Deltaw0,
    dttDeltacposDeltaw0,
    dttDeltac0Deltawpos,
    dttDeltacposDeltawpos
  };

  /** Current status */
  DeltasTestType deltas_test_type_;
  
  /** Log barrier mu in the outer loop. */
  double mu_;

  hiopNlpFormulation* nlp_;

protected: //methods
  /** Decides degeneracy @hess_degenerate_ and @jac_degenerate_ based on @deltas_test_type_ 
   *  when the @num_degen_iters_ > @num_degen_max_iters_
   */
  virtual void update_degeneracy_type();

};

class hiopPDPerturbation : public hiopPDPerturbationBase
{
public:
  /** Default constructor 
   * Provides complete initialization, but uses algorithmic parameters from the Ipopt
   * implementation paper. \ref initialize(hiopNlpFormulation*) should be used to initialize
   * this class based on HiOp option file or HiOp user-supplied runtime options.
   */
  hiopPDPerturbation();
  virtual ~hiopPDPerturbation();

  /** Called when a new linear system is attempted to be factorized 
   */
  virtual bool compute_initial_deltas(hiopVector& delta_wx, 
                                      hiopVector& delta_wd,
                                      hiopVector& delta_cc,
                                      hiopVector& delta_cd);

  /** Method for correcting inertia */
  virtual bool compute_perturb_wrong_inertia(hiopVector& delta_wx, 
                                             hiopVector& delta_wd,
                                             hiopVector& delta_cc,
                                             hiopVector& delta_cd);

  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */
  virtual bool compute_perturb_singularity(hiopVector& delta_wx, 
                                           hiopVector& delta_wd,
                                           hiopVector& delta_cc,
                                           hiopVector& delta_cd);
                                           
protected:
  /** Current and last perturbations, primal is split in x and d, dual in c and d. */
  double delta_wx_curr_db_;
  double delta_wd_curr_db_;
  double delta_cc_curr_db_;
  double delta_cd_curr_db_;

  double delta_wx_last_db_;
  double delta_wd_last_db_;
  double delta_cc_last_db_;
  double delta_cd_last_db_;

protected: // methods
  void set_delta_curr_vec() 
  {
    delta_wx_curr_->setToConstant(delta_wx_curr_db_);
    delta_wd_curr_->setToConstant(delta_wd_curr_db_);
    delta_cc_curr_->setToConstant(delta_cc_curr_db_);
    delta_cd_curr_->setToConstant(delta_cd_curr_db_);
  }

  void set_delta_last_vec() 
  {
    delta_wx_last_->setToConstant(delta_wx_last_db_);
    delta_wd_last_->setToConstant(delta_wd_last_db_);
    delta_cc_last_->setToConstant(delta_cc_last_db_);
    delta_cd_last_->setToConstant(delta_cd_last_db_);
  }

private: //methods
  /** Internal method implementing the computation of delta_w's to correct wrong inertia
   * 
   */
  bool guts_of_compute_perturb_wrong_inertia(double& delta_wx, double& delta_wd);

  double compute_delta_c(const double& mu) const;  

};

class hiopPDPerturbationNormalEqn : public hiopPDPerturbation
{
public:
  hiopPDPerturbationNormalEqn();

  virtual ~hiopPDPerturbationNormalEqn();

  /** Called when a new linear system is attempted to be factorized 
   */
  bool compute_initial_deltas(hiopVector& delta_wx,
                              hiopVector& delta_wd,
                              hiopVector& delta_cc,
                              hiopVector& delta_cd);

  /** Method for correcting inertia */
  bool compute_perturb_wrong_inertia(hiopVector& delta_wx, 
                                     hiopVector& delta_wd,
                                     hiopVector& delta_cc,
                                     hiopVector& delta_cd);
                            
  /** Method for correcting singular Jacobian 
   *  (follows Ipopt closely since the paper seems to be outdated)
   */  
  bool compute_perturb_singularity(hiopVector& delta_wx,
                                   hiopVector& delta_wd,
                                   hiopVector& delta_cc,
                                   hiopVector& delta_cd);

private: //methods
  /** 
   * Internal method implementing the computation of delta_w's to correct wrong inertia
   */
  bool compute_primal_perturb_impl();
                                      
  /** 
   * Internal method implementing the computation of delta_c's to correct wrong inertia
   */
  bool compute_dual_perturb_impl(const double& mu);

protected: //variables
  double delta_c_max_bar_;
  double delta_c_min_bar_;
  double kappa_c_plus_;

};

} //end of namespace
#endif
