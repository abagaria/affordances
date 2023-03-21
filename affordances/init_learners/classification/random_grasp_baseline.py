import random 

class RandomGraspBaseline:
  def __init__(self, grasps):
    self.grasps = grasps 

  def sample(self):
    return random.choice(self.grasps)

    