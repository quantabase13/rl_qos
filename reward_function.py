import numpy as np

"""
produce latency list, throghput list, jitter list and user_preference_matrix acc. to SLA
"""

# each user correspond to a set of  SLA list (latency, throuput, jitter)
#order in list:
# VoIP -> Multimedia streaming -> IOT -> type4

default_latency_SLA_list = [5, 10, 200, 200]
default_throughput_SLA_list = [0.5, 80, 0.1, 0.1]
default_jitter_SLA_list = [1, 1, 10, 10]

# column1(voip)
voip_user_latency_SLA_list = [1, 10, 200, 200] #in ms
voip_user_throughput_SLA_list =[1, 80, 0.1, 0.1]#in mbps
voip_user_jitter_SLA_list = [0.5, 1, 10, 10] #in ms

# column2(MultiMedia streaming)
streaming_user_latency_SLA_list = [5, 5, 200, 200]#in ms
streaming_user_throughput_SLA_list = [0.5, 100, 0.1, 0.1]#in mbps
streaming_user_jitter_SLA_list = [1, 1, 10, 10] #in ms

# column3(IOT)
iot_user_latency_SLA_list = [5, 10, 100, 200] #in ms
iot_user_throughput_SLA_list = [0.5, 80, 0.5, 0.1]#in mbps
iot_user_jitter_SLA_list = [1, 1, 20, 10] #in ms


#each user have a preference matrix
user_preference_matrix = np.random.randint(0, 100, (4, 3)) ## pseudo user preference matrix (for test only)

voip_user_preference_matrix = np.array([\
    [100,100,100],\
    [10,10,10],\
     [1,1,1],\
    [1,1,1]])
streaming_user_preference_matrix = np.array([\
    [10,10,10],\
    [100,100,100],\
    [1,1,1],\
    [1,1,1]])
iot_user_preference_matrix = np.array([\
    [1,1,1],\
    [1,1,1],\
    [100,100,100],\
    [1,1,1]])

class Reward():
    # def __init__(self, latency_list, throughput_list, jitter_list, user_preference):
    def __init__(self):
        self.latency_SLA = iot_user_latency_SLA_list
        self.throghput_SLA = iot_user_throughput_SLA_list
        self.jitter_SLA = iot_user_jitter_SLA_list
        self.user_preference = iot_user_preference_matrix
    def reward_SLA(self,state):
        user_preference_matrix = self.user_preference
        latency_SLA_list = self.latency_SLA
        throughput_SLA_list = self.throghput_SLA
        jitter_SLA_list = self.jitter_SLA
        reward = 0
        for flow_state in state[1:4]:
            throughput = flow_state[0]
            latency = flow_state[1]
            jitter = flow_state[2]
            reward += user_preference_matrix[0][0] *np.exp(-(throughput - throughput_SLA_list[0])**2) + user_preference_matrix[0][1] *np.exp(-(latency - latency_SLA_list[0])**2) + user_preference_matrix[0][2] *np.exp(-(jitter -jitter_SLA_list[0])**2)
        for flow_state in state[4:6]:
            throughput = flow_state[0]
            latency = flow_state[1]
            jitter = flow_state[2]
            reward += user_preference_matrix[1][0] *np.exp(-(throughput - throughput_SLA_list[1])**2) + user_preference_matrix[1][1] *np.exp(-(latency - latency_SLA_list[1])**2) + user_preference_matrix[1][2] *np.exp(-(jitter -jitter_SLA_list[1])**2)
        for flow_state in state[6:16]:
            throughput = flow_state[0]
            latency = flow_state[1]
            jitter = flow_state[2]
            reward += user_preference_matrix[2][0] *np.exp(-(throughput - throughput_SLA_list[2])**2) + user_preference_matrix[2][1] *np.exp(-(latency - latency_SLA_list[2])**2) + user_preference_matrix[2][2] *np.exp(-(jitter -jitter_SLA_list[2])**2)
        for flow_state in state[16:18]:
            throughput = flow_state[0]
            latency = flow_state[1]
            jitter = flow_state[2]
            reward += user_preference_matrix[3][0] *np.exp(-(throughput - throughput_SLA_list[3])**2) + user_preference_matrix[3][1] *np.exp(-(latency - latency_SLA_list[3])**2) + user_preference_matrix[3][2] *np.exp(-(jitter -jitter_SLA_list[3])**2)
        
        return reward

