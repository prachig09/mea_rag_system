package com.mea.rag_app;
import java.time.Duration;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.ollama.OllamaChatModel;

public class MeaRagApp {

    public static void main(String[] args) {
        System.out.println("Initializing Local AI Engine...");

        // 1. point LangChain4j to native Ollama endpoint
        ChatLanguageModel model = OllamaChatModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("phi3.5")
                .temperature(0.0) // Set to 0.0 for deterministic policy answers
                .timeout(Duration.ofMinutes(2)) // Local models take time( boot up on first token)
                .build();

        System.out.println("Sending test prompt to local Phi-3.5 model...");
        
        // 2. Fire a test request
        String response = model.generate("Hello! Confirm you are running locally and state your purpose.");
        
        System.out.println("\n--- Ollama Response ---");
        System.out.println(response);
        System.out.println("-----------------------");
    }
}