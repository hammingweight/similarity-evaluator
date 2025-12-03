package io.github.hammingweight.similarityevaluator;

import java.util.List;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;
import org.springframework.ai.evaluation.Evaluator;

/**
 * An evaluator that computes the cosine similarity between embeddings of expected and actual text.
 * This evaluator is useful for checking the semantic similarity between the expected output and the actual LLM response.
 *
 * @author Carl Meijer (and Qwen3-code).
 */
public class SimilarityEvaluator implements Evaluator {

	/** The embedding model used to generate embeddings for text comparison */
	private final EmbeddingModel embeddingModel;

	/** The minimum cosine similarity threshold required for a pass */
	private final double minimumSimilarity;

	/**
	 * Constructs a SimilarityEvaluator with the default minimum similarity of -1.0.
	 *
	 * @param embeddingModel the embedding model to use for generating text embeddings
	 */
	public SimilarityEvaluator(EmbeddingModel embeddingModel) {
		this(embeddingModel, -1.0);
	}

	/**
	 * Constructs a SimilarityEvaluator with a specified minimum similarity threshold.
	 *
	 * @param embeddingModel the embedding model to use for generating text embeddings
	 * @param minimumSimilarity the minimum cosine similarity required for a pass (between -1.0 and 1.0)
	 * @throws NullPointerException if embeddingModel is null
	 * @throws IllegalArgumentException if minimumSimilarity is outside the range [-1.0, 1.0]
	 */
	public SimilarityEvaluator(EmbeddingModel embeddingModel, double minimumSimilarity) {
		if (embeddingModel == null) {
			throw new NullPointerException("Embedding model cannot be null.");
		}
		if (Math.abs(minimumSimilarity) > 1.0) {
			throw new IllegalArgumentException("Minimum cosine similarity cannot be " + minimumSimilarity + ".");
		}
		this.embeddingModel = embeddingModel;
		this.minimumSimilarity = minimumSimilarity;
	}

	/**
	 * Computes the cosine similarity between two vectors.
	 *
	 * @param vectorA the first vector
	 * @param vectorB the second vector
	 * @return the cosine similarity between the two vectors
	 * @throws AssertionError if vectors have different lengths or if either vector is zero
	 */
	static double cosineSimilarity(float[] vectorA, float[] vectorB) {
		assert vectorA.length == vectorB.length : "Vectors A and B have different lengths.";

		double dotProduct = 0.0;
		double magnitudeA = 0.0;
		double magnitudeB = 0.0;

		for (int i = 0; i < vectorA.length; i++) {
			dotProduct += vectorA[i] * vectorB[i];
			magnitudeA += vectorA[i] * vectorA[i];
			magnitudeB += vectorB[i] * vectorB[i];
		}

		assert magnitudeA != 0.0 : "VectorA is zero.";
		assert magnitudeB != 0.0 : "VectorB is zero.";

		magnitudeA = Math.sqrt(magnitudeA);
		magnitudeB = Math.sqrt(magnitudeB);

		return dotProduct / (magnitudeA * magnitudeB);
	}

	/**
	 * Evaluates the similarity between expected text and actual LLM response using cosine similarity.
	 *
	 * @param evaluationRequest the request containing expected text and LLM response content
	 * @return an EvaluationResponse indicating whether the similarity threshold was met
	 * @throws IllegalArgumentException if the evaluation request contains a data list
	 */
	@Override
	public EvaluationResponse evaluate(EvaluationRequest evaluationRequest) {
		if ((evaluationRequest.getDataList() != null) && (!evaluationRequest.getDataList().isEmpty())) {
			throw new IllegalArgumentException("No data list should be supplied.");
		}

		String expectedText = evaluationRequest.getUserText();
		String llmText = evaluationRequest.getResponseContent();
		EmbeddingResponse embeddingResponse = embeddingModel.embedForResponse(List.of(expectedText, llmText));
		float[] expectedEmbedding = embeddingResponse.getResults().get(0).getOutput();
		float[] actualEmbedding = embeddingResponse.getResults().get(1).getOutput();

		double cosineSimilarity = cosineSimilarity(expectedEmbedding, actualEmbedding);
		return new SimilarityEvaluationResponse(cosineSimilarity >= minimumSimilarity, cosineSimilarity);
	}

	/**
	 * A specialized evaluation response that includes the cosine similarity score.
	 */
	private static class SimilarityEvaluationResponse extends EvaluationResponse {

		/** The cosine similarity score */
		private final double cosineSimilarity;

		/**
		 * Constructs a SimilarityEvaluationResponse.
		 *
		 * @param pass whether the evaluation passed the minimum similarity threshold
		 * @param cosineSimilarity the computed cosine similarity score
		 */
		public SimilarityEvaluationResponse(boolean pass, double cosineSimilarity) {
			super(pass, null, null);
			this.cosineSimilarity = cosineSimilarity;
		}

		/**
		 * Returns the cosine similarity score as a float.
		 *
		 * @return the cosine similarity score
		 */
		@Override
		public float getScore() {
			return (float) cosineSimilarity;
		}
	}

}
