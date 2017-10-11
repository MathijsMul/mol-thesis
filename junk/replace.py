sm_fan_in = self.sm.weight.size(1)
sm_fan_out = self.sm.weight.size(0)
self.sm.weight = xavier_uniform_adapted(self.sm.weight, sm_fan_in, sm_fan_out)
self.sm.bias = xavier_uniform_adapted(self.sm.bias, sm_fan_in, sm_fan_out)
