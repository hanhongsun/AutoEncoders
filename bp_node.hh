/* Author Di Sun */

// This is a research model for building and modify
// efficiency is not the main concern

#ifndef AUTOENCODER_BP_NODE_HH
#define AUTOENCODER_BP_NODE_HH

#include <boost/function.hpp>
#include<math.h>

namespace back_prop
{

namespace // anonymous
{

inline double BPNode::sigmoid_0 (double x) const
{
  return 1.0/(1.0 + exp(-x));
}

inline double BPNode::d_sigmoid_0_d_x (double y) const
{
  return y*(1-y);
}

inline double BPNode::sigmoid_1 (double x) const
{
  double e = exp(2.0 *x);
  return (e - 1.0)/(e + 1.0);
}

inline double BPNode::d_sigmoid_1_d_x (double y) const
{
  return 1 - y*y;
}

inline double BPNode::compute_delta(double error, double out) const
{
  return d_sigmoid_0_d_x(out) * error;
}
  
} // anonymous

class BPNode
{
public:

  //  boost::function<double(double)> sigmoid_fn;
  //  boost::function<double(double)> d_sigmoid_d_x_fn;
  inline double sigmoid_0 (double x) const;
  
  inline double sigmoid_1 (double x) const;
  
  inline double d_sigmoid_0_d_x (double y) const;
  
  inline double d_sigmoid_1_d_x (double y) const;

  inline double compute_delta(double error, double out) const;


  // sort out the acyclic graph if necessary
  virtual void run_forward();

  // run get error
  virtual void run_feedback();
  
  // forward signal
  // sum of input
  double x;

  // output signal
  double out;
  
  // backward 
  //
  double delta;

  double error;

  std::vector<BPNode *> children_nodes;
  
  std::vector<std::vector<double> > W;

  std::vector<double> b;
  
  std::vector<BPNode *> parents_nodes;
};



}//  namespace back_prop
#endif // AUTOENCODER_BP_NODE_HH
