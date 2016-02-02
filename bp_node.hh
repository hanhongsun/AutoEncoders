/* Author Di Sun */

// This is a research model for building and modify
// efficiency is not the main concern

#ifndef AUTOENCODER_BP_NODE_HH
#define AUTOENCODER_BP_NODE_HH

#include <boost/function.hpp>
#include<math.h>
#include<vector>
#include<unordered_map>

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

/**
 * Class BPNode is the Node for back propagation algorithm 
 *
 */
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

  // run time reset for the back function
  // this is supposed to be called by a traverse function
  void back_run_reset();

  // sort out the acyclic graph before hand
  // call the forward_run_reset() before hand
  virtual void run_forward();

  // run get error
  virtual void run_feedback();

  void add_child_simple(BPNode& child);
  
  // forward signal
  // sum of input
  double x_;

  // output signal
  double out_;
  
  // backward 
  //
  double delta_;

  double error_;

  std::vector<BPNode *> children_nodes_;

  // forward Weight
  std::unordered_map<BPNode *, double > W_;

  // bias
  double b_;

  std::unordered_map<BPNode *, double > b_vec_;
  
  std::vector<BPNode *> parents_nodes_;
};



}//  namespace back_prop
#endif // AUTOENCODER_BP_NODE_HH
