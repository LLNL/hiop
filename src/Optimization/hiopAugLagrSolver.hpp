
class hiopAugLagrSolver {

public:
  hiopAugLagrSolver(NLP_CLASS_IN* nlp_in_) 
    : nlp_in(nlp_in)
  {
    inner_nlp = new hiopAugLagrAdapter(nlp_in_);
  }

  //we should make it look as hiopAlgFilterIPM
  //and later have a common parent class

protected:
  
  hiopAugLagrAdapter* inner_nlp;
  NLP_CLASS_IN* nlp_in;
};
