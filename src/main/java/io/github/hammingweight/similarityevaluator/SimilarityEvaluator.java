package io.github.hammingweight.similarityevaluator;

import java.util.List;

import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;
import org.springframework.ai.evaluation.Evaluator;

public class SimilarityEvaluator implements Evaluator {

	private final EmbeddingModel embeddingModel;
	
	private final double minimumSimilarity;
	
	public SimilarityEvaluator(EmbeddingModel embeddingModel, double minimumSimilarity) {
		this.embeddingModel = embeddingModel;
		this.minimumSimilarity = minimumSimilarity;
	}
	
	
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
	
	public static class SimilarityEvaluationResponse extends EvaluationResponse {
		
		private final double cosineSimilarity;
		
		public SimilarityEvaluationResponse(boolean pass, double cosineSimilarity) {
			super(pass, null, null);
			this.cosineSimilarity = cosineSimilarity;
		}
		
		public double getCosineSimilarity() {
			return cosineSimilarity;
		}
	}

}
