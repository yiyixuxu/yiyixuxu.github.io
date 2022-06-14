# It Happened One Frame 
## - demonstrating the amazing power of CLIP model 

I love movies, so as a fun project, I created an app - which you can use [here](https://huggingface.co/spaces/YiYiXu/it-happened-one-frame-2) - that lets you search frames from YouTube videos based on the text you type in. It’s named “It Happened One Frame”, in tribute to the classic 1934 romantic comendy "It Happened One Night".

To use this app, all you need is the link to a Youtube video. For example,  you could search “Macaulay Culkin screams with hands on his cheeks” in a [Home Alone movie clip](https://youtu.be/7EOpoRn9_NA) and get the screenshots that capture the most iconic scene in this classic. 

![Macaulay Culkin screams with hands on his cheeks](/images/Macaulay.png)

This particular image is so popular that you can easily get it from a google search. But with the app, you will be able to search for any scene in any movie as long as you can describe it with words. It is extremely flexible and can handle a wide range of visual concepts: from people, objects and locations, to more abstract features such as emotions, vibes and religions. It recognizes public figures and comic book super heroes. So if you use the actors' name in your search, very likely the app already knows who they are. Even if it does not recognize the name, it will be able to make a reasonably good guess based on the the context of the query, as well as the race and gender of the name - the same way we humans would do. If your movie clip has subtitles, you can even include the content of the dialog in your search, e.g. you can search “Vizzini says inconceivable” in "The Princess Bride".  You can also give specific commands about how the frame was shot, using phrases such as “close-up shot” or  “a overhead shot”. There are so many ways you can formulate your search query. I have had a lot fun using this app to search and revisit some of my favorite movie scenes. The results are astonishingly accurate. It almost feel like magic. I’m going to share some examples with you in this post. But firstly, I want to talk about the incredibly powerful model that made this magic happen - the CLIP model.

### What is CLIP and why did I use it? 

CLIP is a model from Open AI that’s trained with hundreds of millions of images with their captions from the internet. It is trained to learn how semantically-related the texts are to the images. You will find its model architecture surprisingly simple: basically, you put each image and text pair through an image encoder and a text encoder respectively, and bring them into the same N-dimensional embedding space. From there you can calculate the cosine similarity between the two vectors. The objective of learning is to jointly train the image encoder and the text encoder in order to maximize the cosine similarity between the correct image and text pairs, and minimize the cosine similarity between the incorrect pairs at the same time.

A much more difficult question to answer is why CLIP is able to perform so well. Open AI released a second paper that dives deeper into the inner workings of CLIP model via faceted feature visualization. You can read the full paper [here](https://distill.pub/2021/multimodal-neurons/). I will not go into its details in this article. But one amazing finding from this paper is that __CLIP neurons are multimodal__! It means neurons fire to the same concept, regardless of whether it is in photographs, drawings, or text. For example, the “Spiderman” neuron in CLIP will respond to Spiderman in all forms: pictures of Spiderman, comic book drawings of Spiderman, or even images that contain the text “Spiderman”. It means we will be able to leverage subtitles in our movie frame search engine using CLIP. This capability is super handy because often times we want to find a scene based on a line that we remember from the movie. Obviously you can transcribe the audio into text and then perform a keyword search. But CLIP allows you to search for audio and visual content at same time. And it is able to develop a much more comprehensive understanding of the movie frames with sementic information from both the frame itself and the subtitles. 

Natural language supervision allows CLIP to learn a much broader range of visual features when compared with traditional networks. Zero-shot CLIP significantly outperforms Resnet-50 in action recognization. Moreover, it’s able to understand abstract concepts that Resnet could not understand, such as emotions, seasons, and geography. It even understands pop culture, religion and art, as well as concepts in photography such as perspective and magnification. This allows our users to formulate intuitive search queries for practically anything they want to search. 


### Fun with the app
#### Yankee Doodle Dandy 
Let's start with something simple. I want to find a very memorable scene in "Yankee Doodle Dandy". It is toward end of the movie: as James Cagney (pertaining George M. Cohan) leaves the president's office and walks down the stairs in White House, he suddenly start to tap dance and improvises all the way to the bottom, without once looking down his feet. It was really a magnificant scene. I searched "James Cagney dancing down the stairs" in [this clip of Yankee Doodle Dandy's final scene](https://youtu.be/v1rkzUIL8oc). And this is the result 

![James Cagney dancing down the stairs](/images/Cagney.png)

My search query is simple, it includes a person (James Cagney), an object (stairs) and an action (dancing). While it is relatively easy for AI algorithems to identify objects, it's more challenging to __recognize actions__. If you watch the clip, you can tell that Cagney started off by walking down the stairs, and then broke into a the dance at 2:36. But it is not that easy to differenciate dancing from walking when you only look at a single frame. Looking at frames returned by the app, I, as a human, can't confidently say wehther he was dancing or walking in any of the 4 frames. But CLIP knows better - the timestamps (on the upper left cornder of each frame) show that all these 4 frames are captured after 2:36, indicating that Cagney is indeed dancing in these frames. It is quiet impressive! 

#### Once upon a Time in America
I really like [this beautiful scene](https://youtu.be/0diCvgWv_ng) from Once upon a Time in America where the young Deborah (played by Jennifer Connelly) is practicing ballet dance in her parents' restaurant. This clip only contains the the dance sequence but I want to use it as an example to show you how you can search for frames from specific angels and perspectives. 

Let's first see what the resuls looks like when we simply search for "Deborah dancing". You can see that only half of the times Debrah was facing the camera. And all these shots are pretty close-up. 
![Deborah dancing](/images/Deborah-dancing.png")

Let's see what happens if we search "Deborah facing the camera". Voila! I think the app understood me correctly and Deborah is now facing the camera in all 4 frames
![Deborah facing the camera](/images/Deborah-facing.png)

"a wide shot of Deborah practicing her ballet" will get you wide shots:) Now you can see Deborah is wearing a tutu, and she is dancing in a dusty storehouse. Stunning! isn't it? 
![a wide shot of Deborah practicing her ballet(/images/wide-Deborah.png)


#### Paths of Glory
The [ending of Paths of Glory](https://youtu.be/s3ifRA0Kj-8) holds a special place in my heart. It's a movie about war. This particular scene took place in a bar, where a captured German girl was forced to sing for French soldiers. As she started to sing, the rowdy crowd become silent. The soldiers started to hum along and many of them teared up.I vividly remember how the camera move from face to face and the soldiers' facial expressions change. This scene captures humanity in the horror of war and is incredible emotional. It is a perfect example to test the app's ability to identify emotions. 

It understands "rowdy"
![close-up shot of rowdy crowd](/images/rowdy.png)
 
It jointly understand the emotion and action 
![The girl , with tears on her cheeks, begins to sing a song](/images/girl.png)

Wow, not only did it find the frames where audience looks serious, it includes the girl on the stage facing the audience in all 4 frames because I included "listen to her voice" in the search query 
![audience quiets down and listens intently and respectfully to her plaintive voice](/images/voice.png)

It definately understand sadness 
![young man wiping tears](/images/cry1.png)
![a man has tears flowing down his cheeks"](/images/cry2.png)

However it does not seem to understand "sentimental" 
![crowd became sentimental](/images/sentimental.png)

#### Inglourious Basterds

Finally, we are going to play with [this clip of Inglourious basterds](https://youtu.be/rq7qm3T3cPE) - it has subtitles so we can include the content of dialog in our search! 

But first, I want to know if it recognize Brad Pitt - it's an astounding YES! 
![Brad Pitt](/images/BradPitt.png)

I was able to search "Landa sees Bridget's left leg in wrapped in cast and asks her what happened" and find the exact frames where Landa asked Bridget "so what's happened to your lovely leg?" 
![Landa sees Bridget's left leg in wrapped in cast and asks her what happened](/images/leg.png)

If I search "Landa asked Bridget a question", 3 of the 4 frames in the results have a question mark in the subtitles. CLIP's "question" neuron fires when it sees a question mark. 
![Landa asked Bridget a question](/images/question.png)

If I search "she introduce Brad Pitt as Italian stuntman", I got the exact frames where the subtitle says "This is a wonderfurl Italian stuntman, Enzo Gorlomi". There is no way it could have done this without the ability to read subtitles robustly.  
![She introduce Brad Pitt as Italian stuntman](/images/stuntman.png) 

Now let's make it even more fun! what if I search "Bridget introduced Brad Pitt as camera assistant Dominick"? This is a lie - she introduced Omar as the camera assistant, not Brad Pitt. We already knew that the app recognizes Brad Pitt. What would the it do now?  Looks like we've fooled it 
![Bridget introduced Brad Pitts as camera assistant Dominick](/images/Dominick.png)
























