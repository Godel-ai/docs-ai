# Godel.ai

---

# Render.ai : AI Movie Project

- Characters Generation
- Talking Heads
- Scene Graph: Text2Scene
- View Synthesis
- Video Dynamics: DDPAE
- Dialogues
- Voice Synthesis

---

# Samantha.ai : AI Assistant Project

- AI Labs (Virtual Humans and Full Length Creation)
- Replika type Apps: that can go really popular will also be useful
- A person that you can talk as a friend/partner
- Be a good advisor and mentor
- Do therapy and help, be there in loneliness etc
- A philosopher bot, expert bot etc: someone you can discuss philosophy and psychology with

## Nvidia Jarvis + Omniverse = Bot

---

# Fables.ai

- Check the AI dungeon Code and Deployment
- Look at the scene graph generation code for characters and stuff
- Do a literature survey of Story: Visual StoryTelling, Context and Genre specific Dialog Generation
- Create a React Native App of Story Writing Platform like Wattpad
- Make the Visual Storytelling Web tool, Alt-life : life simulator, Pratilipi, Wattpad
- r/WritingPrompts
- r/NoSleep
- The steps for story writing involves: a generated world/scene-setting, characters, character dynamics, keys events, plots, genre and dialogue generation
- Have some characters: and their figures, parse the story into sentences, generate 3D layout and scenes of events, add characters and their interactions in it.
- Grounding, Non-Normative Text and Enhancement of Plots (Events to Sentences), Fine-Tuning of GPT-2 and Generating Stories with Characters with more benefit from different story-writing platforms with others
- So, let us think about the steps story writing involves: a generated world/scene-setting, characters, character dynamics, keys events, plots, genre and dialogue generation
- Fantasy name generator
- The Problem we are trying to Solve : Rich Storytelling Sharing Platform, AI-based Story-Writing Experience
- Market Search - I shared the startups similar yesterday
- Solutions - Web/Mobile Applications, Features
- Business Model - Ad-based etc 

---

# Coga.ai : An AI-based Content Generation Platform

- The full AI Content Generation Circle
- Full Content Creation Automation using AI (Have to make the Circular Diagram)

---

# Emotiv.ai : Real time Voice Cloning

- Text to Speech API with User Quality: Create a Audible/SUNO like App

---

# Philo-Stream : Video Streaming

- TikTok/Bulbul : Short Lecture format, Random Feed based on interests

At a high level:
A rust TTS server hosts two models: a mel inference model and a mel inversion model. The ones I'm using are glow-tts and melgan. They fit together back to back in a pipeline.

Instead of using graphemes, I'm using ARPABET phonemes, and I get these from a lookup table called "CMUdict" from Carnegie Mellon. In the future I'll supplement this with a model that predicts phonemes for missing entries.

Each TTS server only hosts one or two voices due to memory constraints. These models are huge. This fleet is scaled horizontally. A proxy server sits in front and decodes the request and directs it to the appropriate backend based on a ConfigMap that associates a service with the underlying model. Kubernetes is used to wire all of this up.

---

# Open World Game using AI : No Man's Skys
