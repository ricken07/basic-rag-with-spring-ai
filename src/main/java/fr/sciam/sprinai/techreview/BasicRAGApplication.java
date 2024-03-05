package fr.sciam.sprinai.techreview;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.annotation.Bean;

import java.util.Map;

/**
 * @author Ricken Bazolo
 */
@SpringBootApplication
@Slf4j
public class BasicRAGApplication {
    public static void main(String[] args) {
        new SpringApplicationBuilder(BasicRAGApplication.class)
                .web(WebApplicationType.NONE)
                .run(args);
    }
    @Bean
    ApplicationRunner runner(ChatClient chatClient, VectorStore vectorStore) {
        return args -> {

            // Get data and Extract text from page html
            var dataUrl = "https://docs.spring.io/spring-ai/reference/api/etl-pipeline.html";
            var tikaReader = new TikaDocumentReader(dataUrl);
            var extractDocs = tikaReader.get();

            // Split text into chunks
            var tokenTextSplitter = new TokenTextSplitter();
            var chunks = tokenTextSplitter.split(extractDocs.get(0).getContent(), 2048);
            log.info("CHUNKS = "+chunks.size());

            // Create Document for each text chunk
            var documents = chunks.stream()
                    .map(d -> new Document(d, Map.of("name", "etl_pipeline")))
                    .toList();

            // Create embedding for each document and store to vector database
            vectorStore.accept(documents);

            // Retrieve similar chunks from the vector database
            var question = "What is ETL pipeline?";
            var retrievedChunk = vectorStore.similaritySearch(
                    SearchRequest.query("")
                            .withQuery(question) // Text to use for embedding similarity comparison.
                            .withSimilarityThreshold(0.1) // Similarity threshold score to filter the search response by.
                            .withTopK(3) // the top 'k' similar results to return.
            );

            // Combine context and question in a prompt
            var templateTemplate = """
                    Context information is below.
                    RETRIEVED_CHUNK :
                    {retrieved_chunk}
                    Given the context information and not prior knowledge, answer the question.
                    QUESTION: {question}
                    Answer:
                    """;
            var prompt = new PromptTemplate(templateTemplate)
                    .create(Map.of("retrieved_chunk", retrievedChunk, "question", question));

            // Generate response
            var result = chatClient.call(prompt).getResult().getOutput().getContent();
            log.info(result);
        };
    }

}
