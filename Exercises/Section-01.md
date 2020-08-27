> Name: G Reddy Akhil Teja\
> Email: akhilgodvsdemon@gmail.com\
> Date: 26-08-2020

# D2l Exercises

```
1. Which parts of code that you are currently writing could be “learned”, i.e., improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?
```
**Answer:**\
We display images uploaded by user having fixed size. If the image doesn't fit into the dimensions of the box we give an option to crop the image for the fixed size or by default we crop the image from centre position fitting the dimensions.

This part can be improved by learning the focal points in the image from which the images will be cropped by default. For Example faces in an image can considered as the centre positions and then the image can be cropped.

Yes the heuristic design choice as mentioned is to crop the photo from the centre.

```
2. Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using deep learning.
```
**Answer**
- Photo Organizer - Group Photos by faces
- Toggling Home Appliances - Depending on Home Arrivals Time Toggle Home Appliances such as Geyser etc..,.

```
3. Viewing the development of artificial intelligence as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal (what is the fundamental difference)?
```
**Answer**\
The commonalities between these analogies are:
- Coal, Data are Sources and Steam Engine, Algorithms are Consumers
- Steam Engine produces Energy as output and Algorithms produce decisions.

Key Difference:
- Coal is used to power Engines for Energy but Data is used to train Algorithms for decisions.

```
4. Where else can you apply the end-to-end training approach (such as in Fig. 1.1.2)? Physics? Engineering? Econometrics?

```
**Answer**
- Physics: Simulations
- Engineering: Manufacturing Defects detection
- Econometrics: Reinforcement Learning on Taxation. [SalesForce AI Economist](https://www.salesforce.com/company/news-press/stories/2020/4/salesforce-ai-economist/)


# Chapter 1 Summary [D2L Introduction](https://d2l.ai/chapter_introduction/index.html)

Building solutions based on [First Principles](https://fs.blog/2018/04/first-principles/) help in solving complex problems. Still there are few problems that become harder to solve like Problems that are dynamic example: Identifying desired object in an image. A specific object i.e, Rose Flower, can be dynamic in nature, in terms of size, shape, color intensity etc..,. Our scope of creating algorithms are restrictive in terms of our own experience and imagination. Therefore we try to take the help of Machine Learning to help design our systems that learn based on observations that can scale to millions of images where a normal human cannot.

The Key components in ML are:
-   Data
-   Models
-   Objective Functions
-   Optimization Algorithms

## Data:
Without Data we cannot have systems learn. There are different kinds of data that can be used for systems to train such as tabluar data, images, audio, videos, text.

All the data are to be converted to numerical format that can be used to train the systems.

## Models:
Models are the Algorithms that we run on data which inturn learn on the data by adjusting parameters internally. We have various types of Models that are used based on type of requirement and the kind of Data we work on.

## Objective Functions:
These are also called as Cost Function / Loss Function that tell us how extent are our models are deviating from ground truth values. Technically, we strive to achieve global optima in our objective Function.

## Optimization Algorithms:
These help to provide feedback to the Algorithms from Loss Functions by correcting Parameters to achieve global optima.

