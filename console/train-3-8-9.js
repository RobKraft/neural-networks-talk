// 7-segment perceptron training for digits 3, 8, 9
// Run with: node console/train-3-8-9.js

// Segment order: [a b c d e f g]
const digits = {
  3: [1, 1, 1, 1, 0, 0, 1],
  8: [1, 1, 1, 1, 1, 1, 1],
  9: [1, 1, 1, 1, 0, 1, 1]
};

// Training order
const trainingSequence = [3, 8, 9, 3, 9, 8, 3, 8, 9];

// Initialize weights to zero
let weights = {
  3: Array(7).fill(0),
  8: Array(7).fill(0),
  9: Array(7).fill(0)
};

// Utility: dot product
function dot(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

// Training
trainingSequence.forEach((trainDigit, step) => {
  // Compute scores for all digits
  const scores = {};
  for (let d in weights) {
    scores[d] = dot(weights[d], digits[trainDigit]);
  }

  // Find predicted digit
  let predicted = Object.keys(scores).reduce((best, d) => {
    return scores[d] > scores[best] ? d : (scores[d] === scores[best] && d < best ? d : best);
  }, "3");

  // Decide if update is needed
  const updateNeeded = predicted != trainDigit;

  console.log(`Step ${step + 1}: train on ${trainDigit}`);
  console.log(`  Scores: 3=${scores[3]}, 8=${scores[8]}, 9=${scores[9]}`);
  console.log(`  Predicted: ${predicted} ${updateNeeded ? "(wrong, updating)" : "(correct)"}`);

  if (updateNeeded) {
    // Update weights
    const x = digits[trainDigit];
    // Correct digit += x
    weights[trainDigit] = weights[trainDigit].map((w, i) => w + x[i]);
    // Wrong prediction -= x
    weights[predicted] = weights[predicted].map((w, i) => w - x[i]);
  }

  console.log(`  Weights after step:`);
  console.log(`    w3: [${weights[3].join(", ")}]`);
  console.log(`    w8: [${weights[8].join(", ")}]`);
  console.log(`    w9: [${weights[9].join(", ")}]`);
  console.log("------------------------------------------------");
});
