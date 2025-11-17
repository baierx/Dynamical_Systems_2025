#!/usr/bin/env python
# coding: utf-8

# # A Simple Script to Integrate Differential Equations
# ## Function import

# In[3]:


from scipy.integrate   import solve_ivp
from matplotlib.pyplot import subplots


# ## Rate Equation Definition

# In[4]:


def model(t, variables):
    """Harmonic Oscillator"""

    X, Y = variables

    dXdt = - Y
    dYdt = + X

    return [dXdt, dYdt]


# ## Integrate and Plot Result

# In[5]:


initial_conditions = [1, 2]

time_span = (0, 30)

solution = solve_ivp(model, (0, 30), (1, 3), max_step=0.01)

fig, ax = subplots(figsize=(5, 3))

ax.plot(solution.t, solution.y.T);
ax.set_xlabel('Time');


# In[ ]:




