package io.github.hammingweight.similarityevaluator;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class SimilarityEvaluatorTests {

	@Test
	void cosineSimilarity() {
		// Same directions
		double cs = SimilarityEvaluator.cosineSimilarity(new float[] {1.0f}, new float[] {1.0f});
		Assertions.assertEquals(1.0, cs, 0.0001);
		// Opposite directions
		cs = SimilarityEvaluator.cosineSimilarity(new float[] {1.0f}, new float[] {-2.0f});
		Assertions.assertEquals(-1.0, cs, 0.0001);
		// Orthogonal
		cs = SimilarityEvaluator.cosineSimilarity(new float[] {1.0f,0.0f}, new float[] {0.0f,1.0f});
		Assertions.assertEquals(0.0, cs, 0.0001);
		// 45 degree angle
		cs = SimilarityEvaluator.cosineSimilarity(new float[] {1.0f,0.0f}, new float[] {1.0f,1.0f});
		Assertions.assertEquals(0.707, cs, 0.001);
	}

}
