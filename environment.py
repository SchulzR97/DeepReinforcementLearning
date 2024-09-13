import gym
import torch
import cv2 as cv
import numpy as np

class Environment():
    def __init__(self):
        self.steps = 0
        self.state_dim = None
        self.action_dim = None
        self.state = None
        self.done = False

        self.reward_top = 1.
        self.reward_loose = -1.
        self.reward_default = 0.

    def reset(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()
    
    def render(self, display = True, show_debug_info = False):
        raise NotImplementedError()
    
class GymCartPoleV1(Environment):
    def __init__(self):
        super(GymCartPoleV1, self).__init__()
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

    def reset(self) -> tuple[torch.Tensor, float, bool]:
        self.steps = 0
        state, _ = self.env.reset()
        state = torch.tensor(state).unsqueeze(0)
        self.state = state
        self.done = False
        return state, 0., self.done
    
    def step(self, action:int) -> tuple[torch.Tensor, float, bool]:
        state, reward, self.done, _, _ = self.env.step(action)

        self.steps += 1

        if self.done:
            reward = -1.

        state = torch.tensor(state).unsqueeze(0)
        self.state = state
        return state, reward, self.done
    
    def render(self, display = True, show_debug_info = False):
        img = self.env.render()

        if show_debug_info:
            
            
            cart_x = self.state[0, 0].item() / 2.4#4.8
            pole_x = cart_x + self.state[0, 2].item() / 2.4

            top_y = 171
            base_y = 298

            l = top_y - base_y

            c_x = img.shape[1] // 2

            cart_px = int(np.round(c_x + cart_x * img.shape[1] // 2))
            cart_py = base_y


            arrow_len = 30
            # arrow cart velocity
            cart_velocity = self.state[0, 1].item()
            cart_velo_px = int(np.round(cart_px + arrow_len * cart_velocity))
            cart_velo_py = cart_py
            img = cv.arrowedLine(img, (cart_px, cart_py), (cart_velo_px, cart_velo_py), color=(0, 0, 255), thickness=2)

            # arrow pole velocity
            pole_px = int(np.round(c_x + pole_x * img.shape[1] // 2))
            dx = pole_px - cart_px
            theta = np.arccos(dx / l)
            pole_py = int(np.round(cart_py + np.sin(theta) * l))

            pole_velocity = self.state[0, 3].item()
            pole_velo_px = int(np.round(pole_px + np.sin(theta) * arrow_len * pole_velocity))
            pole_velo_py = int(np.round(pole_py - np.cos(theta) * arrow_len * pole_velocity))
            img = cv.arrowedLine(img, (pole_px, pole_py), (pole_velo_px, pole_velo_py), color=(0, 0, 255), thickness=2)

            # text
            img = cv.putText(img, f'Cart Poition: {self.state[0, 0]:0.4f}', (10, 20), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            img = cv.putText(img, f'Cart Velocity: {self.state[0, 1]:0.4f}', (10, 40), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            img = cv.putText(img, f'Pole Position: {self.state[0, 2]:0.4f}', (10, 60), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            img = cv.putText(img, f'Pole Velocity: {self.state[0, 3]:0.4f}', (10, 80), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            img = cv.putText(img, f'Steps: {self.steps}', (10, 100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))
            img = cv.putText(img, f'Theta: {theta / np.pi * 180:0.3f}', (10, 120), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0))

        if display:
            cv.imshow('CartPole-v1', img)
            cv.waitKey(30)
        return img
    
class Pendulum(Environment):
    def __init__(self, l = 1, m = 50., dt = 0.05, damping = 1e-3):
        super(Pendulum, self).__init__()

        self.action_dim = 2
        self.state_dim = 3

        self.l = l
        self.m = m
        self.g = 9.81
        
        self.dt = dt / 1000
        self.damping = damping

        self.factor_omega = 333

        self.reset()

    def reset(self, random = False, theta = 180):
        if random:
            self.px = (-1. + 2. * np.random.random()) * 0.25
            self.theta = np.random.random() * 2 * np.pi
            self.omega = 2 * (np.random.random() - 0.5) * 0.033
        else:
            self.px = 0.
            self.theta = 0. if theta is None else theta / 180 * np.pi
            self.omega = 0.
        self.steps = 0
        
        self.t = 0

        self.done = False
        self.state = torch.tensor([self.px, self.theta, self.omega * self.factor_omega])
        return self.state, 0., self.done
        
    def sim_step(self):
        self.omega *= (1. - self.damping)

        a = np.sin(self.theta) * self.g

        self.omega += a * self.m * self.dt / self.l
        self.theta -= self.omega

        self.t += self.dt
        self.steps += 1
        pass

    def step(self, action:int):
        dv = 100#1e-2#10.
        dx = 100

        domega = dv / self.l

        direction = -1 if action == 0 else 1

        self.px += dx * direction * self.dt

        self.omega += np.cos(self.theta) * domega * direction * self.dt

        self.sim_step()

        self.done = False

        reward_cos = (-np.cos(self.theta)**1) * self.reward_top
        if self.px < -0.5 or self.px > 0.5:
            self.done = True
            reward = self.reward_loose
        # elif -np.cos(self.theta) < 0.8:
        #     self.done = True
        #     reward = self.reward_loose
        else:
            reward = reward_cos * self.reward_top

        self.state = torch.tensor([self.px, self.theta, self.omega * self.factor_omega])
        
        return self.state, reward, self.done

    def render(self, display = True, show_debug_info = False):
        size = (500, 500, 3)
        img = np.full(size, 255, dtype=np.uint8)

        margin = 30
        r = int(np.round(0.05 * img.shape[1]))
        w = (img.shape[1] - 2 * margin)
        h = (img.shape[0] - 2 * margin)

        p1x = img.shape[1] // 2 + int(np.round(self.px * w))
        p1y = img.shape[0] // 2
        p2x = p1x + int(np.round(np.sin(self.theta) * 0.5 * w))
        p2y = p1y + int(np.round(np.cos(self.theta) * 0.5 * h))

        img = cv.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), color=(200, 200, 200), thickness=1)
        img = cv.line(img, (img.shape[1] // 2, img.shape[0] // 2 - 5), (img.shape[1] // 2, img.shape[0] // 2 + 5), color=(200, 200, 200), thickness=1)
        img = cv.rectangle(img, (0, 0), (margin, img.shape[0]), color=(200, 200, 200), thickness=-1)
        img = cv.rectangle(img, (img.shape[1] - margin, 0), (img.shape[1], img.shape[0]), color=(200, 200, 200), thickness=-1)

        cross_size = 5
        img = cv.line(img, (p1x - cross_size, p1y - cross_size), (p1x + cross_size, p1y + cross_size), color=(200,200,200), thickness=2)
        img = cv.line(img, (p1x + cross_size, p1y - cross_size), (p1x - cross_size, p1y + cross_size), color=(200,200,200), thickness=2)

        img = cv.line(img, (p1x, p1y), (p2x, p2y), color=(0,0,0), thickness=1)
        img = cv.circle(img, (p2x, p2y), radius=r, color=(0,0,0), thickness=-1)

        arrow1_x = p2x
        arrow1_y = p2y
        arrow2_x = arrow1_x + int(np.round(np.cos(self.theta + np.pi) * self.omega * 2000))
        arrow2_y = arrow1_y + int(np.round(np.sin(self.theta) * self.omega * 2000))
        img = cv.arrowedLine(img, (arrow1_x, arrow1_y), (arrow2_x, arrow2_y), color=(30, 30, 200), thickness=2)

        if show_debug_info:
            img = cv.putText(img, f'Cart Position: {self.px:0.2f}', (margin + 10, 20), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(60,60,60))
            img = cv.putText(img, f'Angular Velocity: {self.omega:0.2f}', (margin + 10, 40), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(60,60,60))
            img = cv.putText(img, f'Angle: {self.theta:0.2f}', (margin + 10, 60), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(60,60,60))
            img = cv.putText(img, f'Steps: {self.steps}', (margin+10, 80), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(60,60,60))            

        if display:
            cv.imshow('CartPole', img)
            cv.waitKey(30)

        return img

ENVIRONMENT_TYPE = 'CartPole-v1'
CONTROL_MODE = 'random'

if __name__ == '__main__':
    if ENVIRONMENT_TYPE == 'CartPole-v1':
        env = GymCartPoleV1()
    elif ENVIRONMENT_TYPE == 'Pendulum':
        env = Pendulum()
    else:
        raise Exception(f'ENVIRONMENT_TYPE {ENVIRONMENT_TYPE} is not supported.')

    while True:
        state, reward, done = env.reset()
        env.render(display=True, show_debug_info=True)

        while not done:
            if CONTROL_MODE == 'random':
                action = np.random.randint(0, env.action_dim)
            elif CONTROL_MODE == 'manual':
                key = cv.waitKey()
                if key == 2:    # arrow left
                    action = 0
                elif key == 3:  # arrow right
                    action = 1
                else:
                    action = -1
            else:
                raise Exception(f'CONTROL_MODE {CONTROL_MODE} is not supported.')
            
            if action != -1:
                state, reward, done = env.step(action)

            env.render(display=True, show_debug_info=True)

