package com.mea.rag_app;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.List;

public class PolicyIngestionEngine {

    public static void main(String[] args) {
        System.out.println("=== Phase 2: Starting Local Policy Ingestion Engine ===");

        //  local Embedding Engine (Runs inside JVM)
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        //  local Vector Store
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        //  Complex Policy Document
        String complexMeaPolicy = """
                # MINISTRY OF EXTERNAL AFFAIRS POLICY MANUAL (2024-2026)
                
                ## SECTION 1.1: SEMICONDUCTOR & HARDWARE COMPLIANCE
                All tech hardware containing encrypted components coming from the 
                European Union zone must be put in a 30-day quarantine at Mumbai Port until Form 10A is cleared.
                
                ## SECTION 1.2: MARCH 2026 OVERWRITE AMENDMENT
                Under the new 2026 EU-India Digital Corridor agreement, microchips and integrated circuits 
                are officially EXEMPT from physical port quarantines if accompanied by a pre-verified digital certificate.
                This directive completely overrides historical 2024 restrictions for microchip categories.
                """;

        Document document = Document.from(complexMeaPolicy);

        //  Chunking Method for version 0.33.0
        // Slices into segments of 300 characters max, with 30 characters overlap
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("\n[Ingestion] Policy split cleanly into " + segments.size() + " structural data chunks.");

        // transform raw text paragraphs into maths vectors
        System.out.println("[Ingestion] Vectorizing chunks into memory...");
        for (TextSegment segment : segments) {
            Embedding embedding = embeddingModel.embed(segment).content();
            embeddingStore.add(embedding, segment);
        }
        System.out.println("[Ingestion] Indexing complete!");

        System.out.println("\n=== Testing Local Semantic Vector Search ===");
        
        //  Test Query 
        String userQuery = "Are microchips from Germany exempt from the port quarantine?";
        System.out.println("User Query: \"" + userQuery + "\"");

        Embedding queryEmbedding = embeddingModel.embed(userQuery).content();
        List<EmbeddingMatch<TextSegment>> relevantMatches = embeddingStore.findRelevant(queryEmbedding, 1);

        for (EmbeddingMatch<TextSegment> match : relevantMatches) {
            System.out.println("\n[Vector Database Match Confirmed]");
            System.out.println("--------------------------------------------------");
            System.out.println(match.embedded().text());
            System.out.println("--------------------------------------------------");
        }
    }
}