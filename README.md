# TransArt-Multimodal-Application

```
🚀🚀🚀
TransArt seamlessly transcribes audio, translates it into multiple languages, and generates stunning visuals based on the content. Additionally, it offers an interactive chatbot for engaging conversations and image generation, all in one platform.

```

```
<p>
Here are the key steps you followed to build the **TransArt Gradio application**:

1. **Imported Libraries and Set Up Environment**:  
   Imported essential libraries like `gradio`, `whisper`, `GoogleTranslator`, `StableDiffusionXLPipeline`, and others. Set up API keys for Hugging Face and Groq for model integration.

2. **Created Audio Processing Function**:  
   Developed a function (`process_audio`) to handle audio uploads, transcribe speech using Whisper, translate it from Tamil to English using `GoogleTranslator`, and optionally generate an image based on the translation.

3. **Implemented Image Generation from Prompt**:  
   Created a function (`generate_image_from_prompt`) to query a custom Stable Diffusion model, generating images based on user-provided text prompts.

4. **Integrated Chatbot with Image Generation**:  
   Added a chatbot feature (`chatbox`) using the Groq API, where the user enters a prompt and receives both a text-based chatbot response and a generated image.

5. **Designed Gradio Interface**:  
   Built the Gradio app using `gr.Blocks()` with multiple tabs:
   - **Audio to Text Tab**: For audio transcription, translation, and optional image generation.
   - **Prompt to Image Tab**: For generating images directly from user-entered text prompts.
   - **Chatbot Tab**: Allows users to interact with a chatbot and generate images based on prompts.

6. **Custom CSS Styling**:  
   Customized the app’s appearance with CSS, defining background colors, button styles, and text color for a polished user interface.

7. **Launched the Application**:  
   Launched the Gradio app to be hosted on the specified server, making it accessible for live testing and demonstrations.

These steps encapsulate the entire process of developing the **TransArt Gradio application**.
</p>
```

![image](https://github.com/user-attachments/assets/2111f5de-7d51-437c-b6bd-1b1dc495dcbd)


![image](https://github.com/user-attachments/assets/9ae30756-0a6b-4a1f-a3bf-305870e5aa3d)



![image](https://github.com/user-attachments/assets/f8121c06-9c01-4df8-90c4-fa0a3365a044)


![image](https://github.com/user-attachments/assets/34cf547e-0ce2-4468-a346-c4b2ab2c59dd)



![image](https://github.com/user-attachments/assets/107e35b1-be07-4b6c-b3f3-cb0591eec950)

