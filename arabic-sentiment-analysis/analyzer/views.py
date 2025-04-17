import pandas as pd
import torch
import csv
import re
import os
import random
import ollama
import json
import pkg_resources
import language_tool_python
from ar_corrector.corrector import Corrector
from django.shortcuts import render, redirect
from .models import AnalysisResult
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from django.http import HttpResponse
from .twitter_client import fetch_tweets
from django.shortcuts import render
from ar_corrector.corrector import Corrector

# Load Model
model_name = "qwen2.5"

# Context-Based Correction (Ar-Corrector)
corrector = Corrector()

# Arabic LanguageTool
tool = language_tool_python.LanguageTool("ar")  

# Sentiment Mapping
LABELS = ["positive", "neutral", "negative"]

def home(request):
    return render(request, "analyzer/home.html")

# Load Local Dictionary
def load_replacement_dictionary():
    """ Load local dictionary of word replacements """
    replacement_dict = {}
    file_path = "dictionary.txt"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")  # ✅ Extract incorrect → correct mapping
                if len(parts) == 2:
                    incorrect, correct = parts
                    replacement_dict[incorrect] = correct

        print(f"✅ Loaded {len(replacement_dict)} words from dictionary.txt")

    except FileNotFoundError:
        print("⚠️ Warning: dictionary.txt not found. Skipping replacements.")
    except Exception as e:
        print(f"⚠️ Error loading dictionary: {e}")

    return replacement_dict

# Load Local dictionary at startup
replacement_dictionary = load_replacement_dictionary()

# Local Dictionar
def correct_using_local_dictionary(text):
    """ Replace words using local dictionary before any other correction """
    words = text.split()
    corrected_words = [replacement_dictionary.get(word, word) for word in words]  # ✅ Replace words if found
    corrected_text = " ".join(corrected_words)

    print(f"🛠 Dictionary Replacement: {corrected_text}")  # Debugging Output
    return corrected_text

# Manage Local Dictionary
def manage_dictionary(request):
    """ View to manage dictionary words via UI """
    file_path = "dictionary.txt"

    # ✅ Load dictionary into a dictionary object
    replacement_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    incorrect, correct = parts
                    replacement_dict[incorrect] = correct
    except FileNotFoundError:
        pass  # File doesn't exist yet

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "add":
            incorrect_word = request.POST.get("incorrect_word").strip()
            correct_word = request.POST.get("correct_word").strip()
            if incorrect_word and correct_word:
                # ✅ Add new word to dictionary.txt
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(f"{incorrect_word},{correct_word}\n")
                print(f"✅ Added: {incorrect_word} → {correct_word}")

        elif action == "remove":
            word_to_remove = request.POST.get("word_to_remove").strip()
            if word_to_remove in replacement_dict:
                del replacement_dict[word_to_remove]
                # ✅ Rewrite file without removed word
                with open(file_path, "w", encoding="utf-8") as f:
                    for incorrect, correct in replacement_dict.items():
                        f.write(f"{incorrect},{correct}\n")
                print(f"❌ Removed: {word_to_remove}")

        # ✅ Reload dictionary after changes
        return redirect("/manage-dictionary")

    return render(request, "analyzer/manage_dictionary.html", {"dictionary": replacement_dict})

# Ar-Corrector
def correct_spelling_ar_corrector(text):
    """ Use Ar-Corrector for Arabic spelling correction """
    corrected_text = corrector.contextual_correct(text)  # Ar-Corrector
    print(f"🛠 Ar-Corrector Correction: {corrected_text}")  # Debugging
    return corrected_text

# LanguageTool
def correct_spelling_languagetool(text):
    """ Use LanguageTool for Arabic spelling correction """
    matches = tool.check(text)  # ✅ Check for spelling errors
    corrected_text = language_tool_python.utils.correct(text, matches)  # ✅ Apply corrections

    print(f"🛠 LanguageTool Correction: {corrected_text}")  # Debugging Output
    return corrected_text

def analyze_text_with_ollama(text):
    """ Send Arabic text to Ollama for sentiment analysis """

    prompt = f"""
    أنت مساعد لتحليل المشاعر. صنف الجملة التالية إلى واحدة من الفئات التالية:
    - إيجابي
    - سلبي
    - محايد

    فقط اكتب التصنيف بدون أي تفاصيل أخرى.

    الجملة: {text}  
    التصنيف:
    """

    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

    if "message" in response and "content" in response["message"]:
     sentiment = response["message"]["content"].strip()
    else:
     sentiment = "neutral"  # Default fallback


    sentiment_map = {"إيجابي": "positive", "سلبي": "negative", "محايد": "neutral"}
    return sentiment_map.get(sentiment, "neutral"), 90.0  # Assuming static confidence score


def analyze_file(request):
    """ Handle CSV file upload and process sentiment analysis with user-selected spell check options """
    if request.method == 'POST' and request.FILES.get('csv_file'):
        try:
            file = request.FILES['csv_file']

            # Validate file extension
            if not file.name.endswith('.csv'):
                raise ValueError("⚠️ الملف يجب أن يكون بصيغة CSV")

            # ✅ Check which spell checkers are selected
            use_dictionary = request.POST.get("spell_check_dict") == "on"
            use_languagetool = request.POST.get("spell_check_languagetool") == "on"
            use_ar_corrector = request.POST.get("spell_check_ar") == "on"

            print(f"✅ Dictionary Enabled: {use_dictionary}, LanguageTool Enabled: {use_languagetool}, Ar-Corrector Enabled: {use_ar_corrector}")

            # Save uploaded file to media folder
            file_path = os.path.join("media", file.name)
            os.makedirs("media", exist_ok=True)  # Ensure "media" folder exists

            with open(file_path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            # Process CSV based on user selection
            processed_file_path, processed_df = process_csv(file_path, use_ar_corrector, use_languagetool, use_dictionary)

            # Convert to HTML Tables
            original_table = pd.read_csv(file_path).to_html(classes="csv-table", index=False)
            processed_table = processed_df.to_html(classes="csv-table", index=False)

            return render(request, "analyzer/results.html", {
                "original_table": original_table,
                "processed_table": processed_table,
                "processed_data": processed_df.to_dict(orient='records'),
            })

        except Exception as e:
            return render(request, "analyzer/upload.html", {"error": str(e)})

    return render(request, "analyzer/upload.html")

def process_csv(file_path, use_ar_corrector=False, use_languagetool=False, use_dictionary=False):
    df = pd.read_csv(file_path)

    if "text" not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")

    processed_data = []

    for _, row in df.iterrows():
        original_text = row["text"]
        corrected_text = original_text  # Default: No correction

        # ✅ Step 1: Apply dictionary replacements first if enabled
        if use_dictionary:
            corrected_text = correct_using_local_dictionary(original_text)
            print(f"🛠 Dictionary Correction: {corrected_text}")

        # ✅ Step 2: Apply LanguageTool if enabled
        if use_languagetool and corrected_text == original_text:
            corrected_text = correct_spelling_languagetool(corrected_text)
            print(f"🛠 LanguageTool Correction: {corrected_text}")

        # ✅ Step 3: Apply Ar-Corrector if enabled & previous methods didn't fix it
        if use_ar_corrector and corrected_text == original_text:
            corrected_text = correct_spelling_ar_corrector(corrected_text)
            print(f"🛠 Ar-Corrector Correction: {corrected_text}")

        sentiment, confidence = analyze_text_with_ollama(corrected_text)

        processed_data.append({
            "user_id": row.get("user_id", random.randint(1000, 9999)),  # Fallback ID
            "text": corrected_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "platform": row.get("platform", "Unknown"),
            "date": row.get("date", pd.to_datetime("today").strftime('%Y-%m-%d')),
        })

    processed_file_path = "media/latest_processed.csv"
    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(processed_file_path, index=False)

    return processed_file_path, processed_df

def download_processed_csv(request):
    """ Allow user to download the latest processed CSV file """
    file_path = "media/latest_processed.csv"

    # Ensure processed file exists
    if not os.path.exists(file_path):
        return HttpResponse("⚠️ No processed file found. Please analyze a CSV first.", status=404)

    with open(file_path, "rb") as f:
        response = HttpResponse(f.read(), content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="processed_results.csv"'
        return response

def analyze_uploaded_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)

        df["sentiment"] = df["text"].apply(analyze_text_with_ollama)

        # Save processed CSV
        processed_csv_path = os.path.join(settings.MEDIA_ROOT, "processed_results.csv")
        df.to_csv(processed_csv_path, index=False)

        return render(request, 'results.html', {'processed_data': processed_df.to_dict(orient='records')})

def statistics_view(request):
    processed_file_path = os.path.join("media", "latest_processed.csv")

    if not os.path.exists(processed_file_path):
        return render(request, 'analyzer/statistics.html', {"error": "❌ لم يتم العثور على بيانات."})

    df = pd.read_csv(processed_file_path)

    # Count sentiment distribution
    sentiment_counts = df['sentiment'].value_counts().to_dict()

    # Count platform distribution
    platform_counts = df['platform'].value_counts().to_dict() if 'platform' in df.columns else {}

    # Count sentiment per platform (grouped)
    if "platform" in df.columns and "sentiment" in df.columns:
        sentiment_per_platform = df.groupby(["platform", "sentiment"]).size().unstack(fill_value=0)
        sentiment_per_platform = sentiment_per_platform.reset_index().melt(id_vars="platform", var_name="sentiment", value_name="count")
        sentiment_per_platform_list = sentiment_per_platform.to_dict(orient="records")
    else:
        sentiment_per_platform_list = []

    # ✅ Compute **dynamic confidence threshold** using the median
    if "confidence" in df.columns and not df["confidence"].isna().all():
        dynamic_threshold = df["confidence"].median()  # Use median confidence as dynamic threshold
        lower_confidence = df[df["confidence"] < dynamic_threshold].shape[0]
        higher_confidence = df[df["confidence"] >= dynamic_threshold].shape[0]
        confidence_distribution = {
            "low": lower_confidence,
            "high": higher_confidence,
            "threshold": dynamic_threshold  # Pass dynamic threshold to the frontend
        }
    else:
        confidence_distribution = {"low": 0, "high": 0, "threshold": 50}  # Default fallback

    # ✅ Debugging Logs
    print("📊 Confidence Data:", confidence_distribution)

    context = {
        "sentiment_counts": json.dumps(sentiment_counts),
        "platform_counts": json.dumps(platform_counts),
        "sentiment_per_platform": json.dumps(sentiment_per_platform_list),
        "confidence_distribution": json.dumps(confidence_distribution),
        "processed_data": df.to_dict(orient='records'),
    }

    return render(request, 'analyzer/statistics.html', context)

def twitter_analysis(request):
    """ Fetch tweets and analyze sentiment using Ollama """
    if request.method == 'POST':
        hashtag = request.POST.get('hashtag', '')
        tweets = fetch_tweets(hashtag)

        # Clear previous results
        AnalysisResult.objects.all().delete()

        # Process tweets efficiently
        results = []
        for tweet in tweets:
            text = tweet.full_text
            sentiment = analyze_text_with_ollama(text)
            confidence = 95.0  # Placeholder, since Ollama doesn't return probability scores
            results.append(AnalysisResult(text=text, sentiment=sentiment, confidence=confidence, source="Twitter"))

        # Bulk Insert for Performance
        AnalysisResult.objects.bulk_create(results)

        return redirect('results')

    return render(request, 'analyzer/twitter.html')