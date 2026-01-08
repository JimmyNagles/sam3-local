import sys
import requests

def test_segmentation(image_path, prompt):
    url = "http://localhost:8000/segment"
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"prompt_text": prompt}
            
            print(f"Testing segmentation on {image_path} with prompt '{prompt}'...")
            response = requests.post(url, files=files, data=data)
            
            if response.status_code == 200:
                res_json = response.json()
                print("Success!")
                print(f"Status: {res_json.get('status')}")
                results = res_json.get('results', [])
                print(f"Number of results: {len(results)}")
                if results:
                    print(f"First result label: {results[0].get('label')}")
                    print(f"First result confidence: {results[0].get('confidence')}")
                    points = results[0].get('points', [])
                    print(f"First result points count: {len(points)}")
                    print(f"First 5 points: {points[:5]}")
            else:
                print(f"Failed: {response.status_code}")
                print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_client.py <image_path> <prompt>")
        sys.exit(1)
    
    test_segmentation(sys.argv[1], sys.argv[2])
