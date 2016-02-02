/*Author Di Sun*/

#include<>

namespace back_prop
{

BPNode::BPNode()
{}// Nothing other than zeros all default

BPNode::BPNode(std::vector<BPNode *> parents_nodes)
{
  parents_nodes_.reserve(parents_nodes.size());
  for (std::vector<BPNode *>::iterator it = parents_nodes.begin(); it != parents_nodes.end(); ++it)
  {
    if (it == NULL)// should be an error message
      continue;
    (*it)->add_child_simple(this);
    parents_nodes_.push_back(*it);
  }
}

BPNode::add_child_simple(BPNode& child)
{
  children_nodes_.push_back(child);
}


void BPNode::back_run_reset()
{
  // TODO
}
  
void BPNode::run_forward()
{
  x_ = 0;
  for (std::vector<BPNode*>::iterator it = parents_nodes_.begin();
    it != parents_nodes_.end(); ++it)
  {
    x_ += W_[*it]*it->out_;
  }
  x_ += b_;
  out_ = sigmoid_0(x_); // TODO introduce different options
}

  









  
}
