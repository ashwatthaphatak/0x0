#!/bin/bash
# save this as test_modal.sh

set -e  # Exit on error

# Configuration
API_URL="https://akshay-3046--deepfake-defense-web-dev.modal.run"  # UPDATE THIS!

echo "🧪 Starting End-to-End Test..."

# 1. Health check
echo -e "\n1️⃣ Testing health endpoint..."
HEALTH=$(curl -s "$API_URL/health")
echo "Response: $HEALTH"

if echo "$HEALTH" | grep -q "ok"; then
  echo "✅ Health check passed!"
else
  echo "❌ Health check failed!"
  exit 1
fi

# 2. Create test image
#NOT NEEDED -- we will give real image to test


# 3. Submit image
echo -e "\n3️⃣ Submitting image for processing..."
RESPONSE=$(curl -s -X POST "$API_URL/ingest" \
  -F "image=@/Users/akshaydongare/Desktop/sample_ginger_hair.jpg" \
  -F "epsilon=0.05")

JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
echo "Job ID: $JOB_ID"
echo "✅ Image submitted!"

# 4. Poll for completion
echo -e "\n4️⃣ Waiting for processing..."
while true; do
  STATUS=$(curl -s "$API_URL/status/$JOB_ID")
  STATE=$(echo "$STATUS" | jq -r '.status')
  PROGRESS=$(echo "$STATUS" | jq -r '.progress')
  MESSAGE=$(echo "$STATUS" | jq -r '.message')
  
  echo "   Status: $STATE | Progress: $PROGRESS% | $MESSAGE"
  
  if [ "$STATE" = "complete" ]; then
    echo "✅ Processing complete!"
    break
  elif [ "$STATE" = "failed" ]; then
    echo "❌ Processing failed!"
    echo "$STATUS" | jq
    exit 1
  fi
  
  sleep 2
done

# 5. Download result
echo -e "\n5️⃣ Downloading result..."
curl -s "$API_URL/status/$JOB_ID" | \
  jq -r '.result_url' | \
  sed 's/data:image\/png;base64,//' | \
  base64 -d > vaccinated_image.png

SCORE=$(echo "$STATUS" | jq -r '.score')
echo "✅ Protection score: $SCORE"
echo "✅ Saved to vaccinated_image.png"

# 6. Verify file
if [ -f vaccinated_image.png ]; then
  SIZE=$(wc -c < vaccinated_image.png)
  echo "✅ Output file size: $SIZE bytes"
else
  echo "❌ Output file not created!"
  exit 1
fi

echo -e "\n🎉 All tests passed!"