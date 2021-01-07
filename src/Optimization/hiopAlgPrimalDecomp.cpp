//include header file

#include "hiopAlgPrimalDecomp.hpp"

#include "hiopInterfacePrimalDecomp.hpp"


#include <cmath>
#include <cstring>
#include <cassert>

namespace hiop
{

  double hiopAlgPrimalDecomposition::getObjective() const
  {
    return master_prob_->get_objective();
  }

  void hiopAlgPrimalDecomposition::getSolution(double* x) const
  {
    master_prob_->get_solution(x);
  }
  
  void hiopAlgPrimalDecomposition::getDualSolutions(double* zl, double* zu, double* lambda)
  {
    assert(false && "not implemented");
  }

  inline hiopSolveStatus hiopAlgPrimalDecomposition::getSolveStatus() const
  {
    return solver_status_;
  }
  int hiopAlgPrimalDecomposition::getNumIterations() const
  {
    assert(false && "not yet implemented");
    return 9;
  }
  hiopSolveStatus hiopAlgPrimalDecomposition::run()
  {

    if(comm_size_==1)
    {
      return run_single();
    }

    printf("total ranks %d\n",comm_size_); //printf("my rank starting 1 %d)\n",my_rank_);
    //initial point = ?
    //for now set to all zero
    for(int i=0; i<n_; i++) {
      x_[i] = 0.;
    }
      
    bool bret;
    int rank_master=0; //master rank is also the solver rank
    //Define the values and gradients as needed in the master rank
    double rval = 0.;
    double grad_r[n_];
    for(int i=0; i<n_; i++) {
      grad_r[i] = 0.;
    }
    //local recourse terms for each evaluator, but defined accross all processors
    double rec_val = 0.;
    double grad_acc[n_];
    for(int i=0; i<n_; i++) grad_acc[i] = 0.;
    //this is an example of the usage of the classes' design and has almost nothing to do with
    //the actual algorithm (master rank does all the computations)

    //hess_appx_2 has to be declared by all ranks while only rank 0 uses it
    HessianBBApprox*  hess_appx_2 = new HessianBBApprox(n_);



    // Outer loop starts
    for(int it=0; it<max_iter;it++){

      printf("my rank iteration  %d)\n", my_rank_);
      // solve the base case
      if(my_rank_ == 0 && it==0) //initial solve
      { 
	printf("my rank for solver  %d)\n", my_rank_);
        //solve master problem base case(solver rank supposed to do it)
	

        solver_status_ = master_prob_->solve_master(x_,false);
        printf("solve master done  %d)\n", my_rank_);
        // to do, what if solve fails?
        if(solver_status_){     

        }
	for(int i=0;i<n_;i++) printf("%d x %18.12e\n",i,x_[i]);

      }

      // send base case solutions to all ranks
      //todo error control
      int ierr = MPI_Bcast(x_, n_, MPI_DOUBLE, rank_master, comm_world_);
      assert(ierr == MPI_SUCCESS);



      //assert("for debugging" && false); //for debugging purpose
      // set up recourse problem send/recv interface
      std::vector<ReqRecourseApprox* > rec_prob(comm_size_);
      for(int r=0; r<comm_size_;r++)
      {
        rec_prob[r] = new ReqRecourseApprox(n_);
      }
      ReqContingencyIdx* req_cont_idx = new ReqContingencyIdx(0);

      // master part
      if(my_rank_ == 0)
      {
        // array for number of indices, this is subjected to change	
        rval = 0.;
        for(int i=0; i<n_; i++)
	{
          grad_r[i] = 0.;
        }
        int* cont_idx = new int[S_];
        for(int i=0;i<S_;i++)
	{
 	  cont_idx[i]=i;
        }
        // The number of contigencies should be larger than the number of processors
        // Otherwise not implemented
        assert(S_>=comm_size_-1);
        // idx is the next contingency to be sent out from the master
        int idx = 0;
        // Initilize the recourse communication by sending indices to the 
        // Using Blocking send here
        for(int r=1; r< comm_size_;r++)
	{
          int cur_idx = cont_idx[idx];
          int ierr = MPI_Send(&cur_idx, 1, MPI_INT, r, 1,comm_world_);
          assert(MPI_SUCCESS == ierr);  
          //printf("rank %d to get contingency index  %d\n", r,cur_idx);
	  idx += 1;
        }
        int mpi_test_flag; // for testing if the send/recv is completed
        // Posting initial receive of recourse solutions from evaluators
        for(int r=1; r< comm_size_;r++)
	{
          //int cur_idx = cont_idx[idx];
	  rec_prob[r]->post_recv(2,r,comm_world_);// 2 is the tag, r is the rank source 
          //printf("receive flag for contingency value %d)\n", mpi_test_flag);
        }	
        // both finish_flag and last_loop are used to deal with the final round remaining contingencies.
        // Some ranks are finished while others are not. The loop needs to continue to fetch the results. 
        std::vector<int> finish_flag(comm_size_);
        for(int i=0;i<comm_size_;i++)
	{
	  finish_flag[i]=0;
	}
        int last_loop = 0;
        while(idx<=S_ || last_loop)
	{ 
          //std::this_thread::sleep_for(std::chrono::milliseconds(10)); //optional, used to adjust time
          for(int r=1; r< comm_size_;r++){
            //int cur_idx = cont_idx[idx];
            //int ierr = MPI_Isend(&rec_val, 1, MPI_DOUBLE, rank_master, 2, comm_world_, &request_[my_rank_]);
            int mpi_test_flag = rec_prob[r]->test();
            if(mpi_test_flag && (finish_flag[r]==0))// receive completed
	    { 
              printf("idx %d sent to rank %d\n", idx,r);

	      // add to the master rank variables
              rval += rec_prob[r]->value();
              for(int i=0;i<n_;i++)
	      {
	        grad_r[i] += rec_prob[r]->grad(i);
	      }
	      if(last_loop){
	        finish_flag[r]=1;
	      }
              // this is for dealing with the end of contingencies where some ranks have already finished
	      if(idx<S_)
	      {
	        req_cont_idx->set_idx(cont_idx[idx]);
	        req_cont_idx->post_send(1,r,comm_world_);
                //int cur_idx = cont_idx[idx];
	        // sending the next contingency index
	        //ierr = MPI_Isend(&cur_idx, 1, MPI_INT, r, 1, comm_world_, &request_[0+r*4]);	
	        rec_prob[r]->post_recv(2,r,comm_world_);// 2 is the tag, r is the rank source 
                //printf("recourse value: is %18.12e)\n", rec_prob[r]->value());
	      }else{
	        finish_flag[r] = 1;
	        last_loop = 1; 
	      }
	      idx += 1; 
	    } 
          }
	  // Current way of ending the loop while accounting for all the last round of results
	  if(last_loop){
	    last_loop=0;
            for(int r=1; r< comm_size_;r++){
	      if(finish_flag[r]==0){last_loop=1;}
	    }
	  }

        }
        // send end signal to all evaluators
        int cur_idx = -1;
        for(int r=1; r< comm_size_;r++)
	{
	  req_cont_idx->set_idx(-1);
	  req_cont_idx->post_send(1,r,comm_world_);
          //int ierr = MPI_Isend(&cur_idx, 1, MPI_INT, r, 1, comm_world_, &request_[0+r*4]);	
        }

      }

      //
      //workers
      if(my_rank_ != 0)
      {
        //int cpr = S_/(comm_size_-1); //contingency per rank
        //int cr = S_%(comm_size_-1); //contingency remained
        //printf("my rank start evaluating work %d)\n",my_rank_);
        std::vector<int> cont_idx(1); // currently sending/receiving one contingency index at a time
        int cont_i = 0;
        cont_idx[0] = 0;
        //int cur_idx = 0;
        //Receive
        int mpi_test_flag = 0;

        int ierr = MPI_Recv(&cont_i, 1, MPI_INT, rank_master, 1,comm_world_, &status_);
        assert(MPI_SUCCESS == ierr);  
        cont_idx[0] = cont_i;
        //printf("contingency index %d, rank %d)\n",cont_idx[0],my_rank_);
        // compute the recourse function values and gradients
        rec_val = 0.;
        for(int i=0; i<n_; i++)
	{
	  grad_acc[i] = 0.;
	}
        double aux=0.;
        for(int ri=0; ri<cont_idx.size(); ri++)
	{
          aux = 0.;
          int idx_temp = cont_idx[ri];
          bret = master_prob_->eval_f_rterm(idx_temp, n_, x_, aux); //need to add extra time here
          if(!bret)
	  {
              //todo
          }
          rec_val += aux;
        }
        //printf("recourse value: is %18.12e)\n", rec_val);
        double grad_aux[n_];
        for(int ri=0; ri<cont_idx.size(); ri++)
	{
          int idx_temp = cont_idx[ri];
          bret = master_prob_->eval_grad_rterm(idx_temp, n_, x_, grad_aux);
          if(!bret)
          {
            //todo
          }
          for(int i=0; i<n_; i++)
            grad_acc[i] += grad_aux[i];
        }
        rec_prob[my_rank_]->set_value(rec_val);
        rec_prob[my_rank_]->set_grad(grad_acc);
        rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);

        // post receive to the next index
        //ierr = MPI_Test(&request_[0], &mpi_test_flag, &status_);
        //ierr = MPI_Irecv(&cont_idx[0], 1, MPI_INT, rank_master, 1, comm_world_, &request_[0]); 
        req_cont_idx->post_recv(1, rank_master, comm_world_);

        while(cont_idx[0]!=-1)
	{
          //printf("contingency index %d, rank %d)\n",cont_idx[0],my_rank_);
	  //mpi_test_flag = rec_prob[my_rank_]->test();
          //ierr = MPI_Test(&request_[0], &mpi_test_flag, &status_);
          mpi_test_flag = req_cont_idx->test();
          // contigency starts at 0 
          // sychronous implmentation of contingencist
          /*                   
          std::vector<int> cont_idx;
          if(my_rank_==comm_size_-1){
            for(int i=0;i<cpr+cr;i++){
	      int idx_temp = i+(my_rank_-1)*cpr;
              cont_idx.push_back(idx_temp);  //currently the last one gets the most contingency, not optimal
	    }
          }
          else{
	    for(int i=0;i<cpr;i++){
	      int idx_temp = i+(my_rank_-1)*cpr;
              cont_idx.push_back(idx_temp);
	    }
          }
          */
          if(mpi_test_flag)
	  {
            //printf("contingency index %d, rank %d)\n",cont_idx[0],my_rank_);
            for(int ri=0; ri<cont_idx.size(); ri++)
	    {
              cont_idx[ri] = req_cont_idx->value();
	    }
            if(cont_idx[0]==-1)
	    {
	      break;
	    }
            rec_val = 0.;
            for(int i=0; i<n_; i++)
	    {
	      grad_acc[i] = 0.;
	    }
            double aux=0.;

            //assert("for debugging" && false); //for debugging purpose
            for(int ri=0; ri<cont_idx.size(); ri++)
	    {
              aux = 0.;
              int idx_temp = cont_idx[ri];
              bret = master_prob_->eval_f_rterm(idx_temp, n_, x_, aux); //need to add extra time here
              if(!bret)
	      {
              //todo
              }
              rec_val += aux;
            }
            //printf("recourse value: is %18.12e)\n", rec_val);
            double grad_aux[n_];
            for(int ri=0; ri<cont_idx.size(); ri++)
	    {
              int idx_temp = cont_idx[ri];
              bret = master_prob_->eval_grad_rterm(idx_temp, n_, x_, grad_aux);
              if(!bret)
              {
                //todo
              }
              for(int i=0; i<n_; i++)
	      {
                grad_acc[i] += grad_aux[i];
	      }
            }

            rec_prob[my_rank_]->set_value(rec_val);
	    rec_prob[my_rank_]->set_grad(grad_acc);
            rec_prob[my_rank_]->post_send(2, rank_master, comm_world_);
            //do something with the func eval and gradient to determine the quadratic regularization  
            //printf("send recourse value flag for test %d \n", mpi_test_flag);
        
	    //post recv for new index
            req_cont_idx->post_recv(1, rank_master, comm_world_);
	    //ierr = MPI_Irecv(&cont_idx[0], 1, MPI_INT, rank_master, 1, comm_world_, &request_[0]); 	  
          }
          //std::this_thread::sleep_for(std::chrono::milliseconds(50));
          //assert("for debugging" && false); //for debugging purpose
        }
      }

      //int err= MPI_Reduce(&rec_val,&rval,1, MPI_DOUBLE, MPI_SUM, rank_master, comm_world_);
      //err= MPI_Reduce(&grad_acc,&grad_r,n_, MPI_DOUBLE, MPI_SUM, rank_master, comm_world_);
      //assert(err == MPI_SUCCESS);
      if(my_rank_==0)
      {
        //std::this_thread::sleep_for(std::chrono::milliseconds(100));
        //assert("for debugging" && false); //for debugging purpose

        MPI_Status mpi_status; 

        double hess_appx[n_]; //Hessian is computed on the solver/master
        for(int i=0; i<n_; i++) hess_appx[i] = 1.0;
     
        if(it==0){
          hess_appx_2->initialize(x_, grad_r);
	}else{
          hess_appx_2->update_hess_coeff(x_, grad_r, rval);

	  double alp_temp = hess_appx_2->get_alpha();
          printf("alpha %18.12e\n",alp_temp);
          double convg = hess_appx_2->check_convergence(grad_r);
          printf("convergence measure %18.12e\n",convg);
	}

        // wait for the sending/receiving to finish
        int mpi_test_flag = 0;
        for(int r=1; r<comm_size_;r++)
	{
          //int ierr = MPI_Wait(&request_[r*4+0], &status_);
          MPI_Wait(&(rec_prob[r]->request_), &status_);
          MPI_Wait(&req_cont_idx->request_, &status_);
          //ierr = MPI_Wait(&request_[r*4+2], &status_);
        }
        // for debugging purpose print out the recourse gradient
        for(int i=0;i<n_;i++) 
	{
	  printf("%d grad %18.12e\n",i,grad_r[i]);
	}
	//todo S_ doesn't have to be bigger than n_ now right?
        RecourseApproxEvaluator* evaluator = new RecourseApproxEvaluator(n_,S_,rval,grad_r, 
		                                hess_appx, x_);
        for(int i=0;i<n_;i++) 
	{
	  printf("%d x0 %18.12e\n",i,x_[i]);
	}
        bret = master_prob_->set_recourse_approx_evaluator(n_, evaluator);
        //bret = master_prob_->set_quadratic_regularization(RecourseApproxEvaluator* evaluator);
        if(!bret)
        {
          //todo
        }
        solver_status_ = master_prob_->solve_master(x_,true);
        delete[] evaluator;
      }
      else{
        std::this_thread::sleep_for(std::chrono::milliseconds(100));    
      }
    }
    if(my_rank_==0)
    {
      return solver_status_;
    }else{
      return Solve_Success;    
    }
  }


  hiopSolveStatus hiopAlgPrimalDecomposition::run_single()
  {
    //only one rank

    //initial point = ?
    //for now set to all zero
    for(int i=0; i<n_; i++) {
      x_[i] = 0.;
    }
      
    bool bret;
    int rank_master=0; //master rank is also the solver rank
    //Define the values and gradients as needed in the master rank
    double rval = 0.;
    double grad_r[n_];
    for(int i=0; i<n_; i++) {
      grad_r[i] = 0.;
    }

    //hess_appx_2 has to be declared by all ranks while only rank 0 uses it
    HessianBBApprox*  hess_appx_2 = new HessianBBApprox(n_);

    // Outer loop starts
    for(int it=0; it<max_iter;it++){

      // solve the base case
      if(it==0) //initial solve
      { //printf("my rank for solver  %d)\n", my_rank_);
        //solve master problem base case(solver rank supposed to do it)
        solver_status_ = master_prob_->solve_master(x_,false);
        // to do, what if solve fails?
        if(solver_status_){     

        }
      }

      // array for number of indices, this is subjected to change	
      rval = 0.;
      for(int i=0; i<n_; i++)
      {
        grad_r[i] = 0.;
      }
      int* cont_idx = new int[S_];
      for(int i=0;i<S_;i++)
      {
        cont_idx[i]=i;
      }
      // The number of contigencies should be larger than the number of processors, which is 1
      // idx is the next contingency to be sent out from the master
      int idx = 0;
        
        
      for(int i=0; i< S_;i++){
        int idx_temp = cont_idx[i];
	double aux=0.;
        bret = master_prob_->eval_f_rterm(idx_temp, n_, x_, aux); //need to add extra time here
        if(!bret)
        {
              //todo
        }
        rval += aux;
        //assert("for debugging" && false); //for debugging purpose
      
        double grad_aux[n_];
        bret = master_prob_->eval_grad_rterm(idx_temp, n_, x_, grad_aux);
        if(!bret)
        {
            //todo
        }
        for(int i=0; i<n_; i++)
	{
          grad_r[i] += grad_aux[i];
	}
      }        
      double hess_appx[n_]; //Hessian is computed on the solver/master
      for(int i=0; i<n_; i++) hess_appx[i] = 1.0;
     
      if(it==0){
        hess_appx_2->initialize(x_, grad_r);
      }else{
        hess_appx_2->update_hess_coeff(x_, grad_r, rval);

	double alp_temp = hess_appx_2->get_alpha();
        printf("alpha %18.12e\n",alp_temp);
        double convg = hess_appx_2->check_convergence(grad_r);
        printf("convergence measure %18.12e\n",convg);
      }

      // for debugging purpose print out the recourse gradient
      for(int i=0;i<n_;i++) 
      {
        printf("%d grad %18.12e\n",i,grad_r[i]);
      }
      RecourseApproxEvaluator* evaluator = new RecourseApproxEvaluator(n_,S_,rval,grad_r, 
		                                hess_appx, x_);
      for(int i=0;i<n_;i++) 
      {
	printf("%d x0 %18.12e\n",i,x_[i]);
      }
      bret = master_prob_->set_recourse_approx_evaluator(n_, evaluator);
      //bret = master_prob_->set_quadratic_regularization(RecourseApproxEvaluator* evaluator);
      if(!bret)
      {
        //todo
      }
      solver_status_ = master_prob_->solve_master(x_,true);
      delete[] evaluator;
    }
      return Solve_Success;    
  }

}//end namespace
