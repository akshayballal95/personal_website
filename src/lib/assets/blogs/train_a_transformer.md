---
title: How to Train your Transformer
author: Akshay Ballal
stage: live
image: https://res.cloudinary.com/practicaldev/image/fetch/s--EfkHDr34--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_66%2Cw_800/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/y6thhqvp9jthxcey71he.gif
description: Learn the tricks used to train transformer models. I introduce warmup, pretraining, and fine-tuning
date: 12/18/2023
---

Dragons are cool. They are mythical; they can fly, and they can breathe fire. Transformers are no less of a dragon in the world of AI. You need to tame them to fly them. And trust me, they need a lot of taming and caring, probably more than an actual dragon. If you do it wrong, you might as well end up like Ash Ketchum. It’s part art, part science. 

![Ash Ketchum burnt by Charizard](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/y6thhqvp9jthxcey71he.gif)

These are the steps you can use to train your transformer better. 

## 1. Don’t train a Transformer

Yes, you heard it right. Transformers are not the best for all AI tasks. You are sometimes better off with another tried and tested method. For example, I was putting together a network for supervised learning to play games and I had serious problems with getting a transformer to work. That led me down a rabbit hole of trying to understand what works and does not work with Transformers. Finally, I ended up using 3D Convolutions with ResNet 3D and got much better results (article on this coming soon). If temporal relationships are what you want to capture, then earlier methods like LSTM and GRUs still work well. These are what I found are the main drawbacks of Transformers.

1. **They are data-hungry.** Transformers without enough data are like dragons without wings. If you think that your AI can do well with X amount of data and you plan to use Transformers, you better collect at least 2X or, even better, 3X amount of data. This was pointed out in the original paper on Transformers, where the author says that this has to do with Transformers not having enough inductive bias. Let me know in the comments if you know exactly what this means. 

2. **They can overfit very fast.** This relates to the first point: having more data is a solution. This is why only big corporations have been able to tame transformers for any reasonable tasks. But with Mistral coming along, that's changing, but still, they had a $100M to start with. Academic researchers and students don't have that sort of resources.

3. **High Computational Demands.** With great data comes great data crunching. So you better have the latest Nvidia Chips at your disposal. Any good transformer network that can produce passable results has at least 100 million parameters, which you must process across a lot of data. It is quite some heavy lifting. No wonder why dragons are so big. 

But it's not the end. Transformers still have a chance. With some clever tricks, you can actually tame them. But I wanted to give you guys a heads-up and say that Transformers are not magical and need a lot more work than you would think to get working. If you are willing to put in the effort and transformer is the only viable option left for you, then read on. 

## 2. Warm-up

![Dragon Running](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/11rsmdxpgcu5oa14lqxn.png)

Just as a dragon requires a warm-up before embarking on a journey, so does your Transformer model. In the realm of AI, warm-up refers to the initial phase of training where the learning rate is gradually increased. This helps the model converge to optimal parameters more effectively. Skipping the warm-up phase can lead to suboptimal performance, akin to attempting to ride a dragon without allowing it to stretch its wings and prepare for flight.

It's important to note that the warm-up phase sets the stage for successful fine-tuning. During this period, the model adapts to the specific characteristics of your dataset, ensuring that it can navigate the challenges unique to your application.

Hidden in the appendices of some literature, you will find tables like this


![ViT Paper](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/xx81pdpw8ljwubanqs0b.png)

From “Images are worth 16x16 Words”. A paper on transformers. 

What you do in warm-up is this. Let’s say you want to train your network at 8 x 10^-4 learning rate. But if you start at the start of the training, your gradients are huge and your weights change drastically. So, you want to start at a very low learning rate and gradually increase up to your base learning rate. This increase happens over a certain number of time steps. 

## 3. Pre-training is your Friend


![Dragon in Gym](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ylnw5fflsijdrvqyq56g.png)

When dealing with Transformers, pre-training is a crucial step. Just like a dragon needs to learn to spread its wings before taking flight, Transformers benefit from pre-training on vast amounts of data. This process involves exposing the model to a diverse range of tasks and data, allowing it to grasp the intricacies of language and context. Pre-training provides the model with a solid foundation, much like teaching a dragon to understand its surroundings before venturing into the skies. 

Here is where your bottomless well of data comes in handy. One option is to use pre-trained weights and completely skip this step. If you are reusing a network and believe that your data closely resembles the original dataset the network was trained on, then you are lucky, and someone else has done the hard work for you. You can move to the next step. For example, if your application involves classifying images, rather than setting up the complete network from convolution to Multi-head Self Attention, you can find off-the-shelf Vision Transformer Models and use pre-trained weights.

## 4. Fine-tuning is your Best Friend


![Love your Dragon](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/k2wigo3jgmth1hm3d1gl.png)

While pre-training lays the groundwork, fine-tuning tailors the model to your specific needs. Think of it as customizing a dragon's training regimen to suit your riding preferences. Fine-tuning allows you to adapt the pre-trained Transformer to your domain, whether it's image recognition, natural language processing, or any other AI task. This step ensures that the model becomes specialized and performs optimally in the context of your unique requirements.

But here's the catch. Transformer models come with their own Fine Tuning strategies. You can find them hidden away in papers or you can just experiment. But this is what generally works. 

1. Have enough data (yeah I know, I have been saying data a lot).
2. Use a learning rate scheduler to lower your learning rate as the fine-tuning progresses through epochs.
3. Check your validation score at every epoch to make sure you are not overfitting.
4. Use some form of data augmentation technique depending on your domain. For images, trivial augmentation works quite well. There are sentence augmentation techniques as well. 

In conclusion, training a transformer model can be a challenging task that requires careful planning and execution. Transformers are not always the best option, and you may need to resort to other methods depending on your specific AI task. However, if you decide to train a transformer, remember to warm up your model, pre-train it on vast amounts of data, and fine-tune it to adapt to your domain. With these steps, you can successfully tame your AI dragon and achieve optimal performance. If you have more tips and tricks, do share them in the comments.

---

Want to connect?

🌍[My Website](http://www.akshaymakes.com)  
🐦[My Twitter](https://twitter.com/akshayballal95)  
👨[My LinkedIn](https://www.linkedin.com/in/akshay-ballal/)