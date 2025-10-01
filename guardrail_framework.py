"""
SecureLife Insurance Guardrail Detective Framework

A framework for building and testing input guardrails for conversational AI systems.
Designed for the TechDays Guardrails Tutorial.
"""

import cohere
import openai
import pandas as pd
import time
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm


class GuardrailDetective:
    """
    A framework for building and testing guardrails for SecureLife Insurance's chatbot.
    
    Your mission: Protect the customer support chatbot by building a robust input guardrail
    that allows legitimate insurance questions while blocking everything else.
    
    The company policy is complex and nuanced - you'll need to discover it through
    investigation and iteration!
    """
    
    def __init__(self, api_key: str, system_prompt: str = None, batch_size: int = 50, provider: str = "cohere"):
        """
        Initialize the Guardrail Detective framework.
        
        Args:
            api_key: Your API key (Cohere or OpenAI)
            system_prompt: Your custom guardrail prompt (if None, uses placeholder)
            batch_size: Number of examples to process per batch (to handle rate limits)
            provider: LLM provider to use ("cohere" or "openai")
        """
        self.provider = provider.lower()
        self.batch_size = batch_size
        self.company_name = "SecureLife Insurance Company"
        self.system_prompt = system_prompt
        
        # Initialize client and model based on provider
        if self.provider == "cohere":
            self.client = cohere.ClientV2(api_key=api_key)
            self.model_id = 'command-r-plus-08-2024'
        elif self.provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
            self.model_id = 'gpt-4o'
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'cohere' or 'openai'")
        
        # Initialize Guardrails AI Guard 
        # TODO: If you want to use Guardrails AI, uncomment the line below and follow the instructions in the _initialize_guard method
        # self.guard = self._initialize_guard()
    
    def _chat(self, system_prompt: str, user_message: str) -> str:
        """
        Unified chat method that works with both Cohere and OpenAI.
        
        Args:
            system_prompt: The system prompt
            user_message: The user message
            
        Returns:
            The response text from the LLM
        """
        if self.provider == "cohere":
            response = self.client.chat(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.message.content[0].text.strip()
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content.strip()

    def run_batch_guardrail(self, user_inputs: List[str]) -> List[str]:
        """
        Process multiple inputs using all three guardrail types.
        Uses batch processing for LLM calls to save costs, and applies
        custom and off-the-shelf guardrails individually to each input.
        
        Args:
            user_inputs: List of user messages to classify
            
        Returns:
            List of classifications ("ALLOW" or "BLOCK") for each input
        """
        if not user_inputs:
            return []
        
        # Get LLM results for the batch
        llm_results = self._llm_batch_classify(user_inputs)
        
        # Apply custom guardrails to each input individually
        custom_results = [self.custom_guardrails(user_input) for user_input in user_inputs]
        
        # Combine results (block if any says block)
        final_results = []
        for llm_result, custom_result in zip(llm_results, custom_results):
            if llm_result == "BLOCK" or custom_result == "BLOCK":
                final_results.append("BLOCK")
            else:
                final_results.append("ALLOW")
        
        return final_results
    
    def llm_guardrails(self, user_input: str) -> str:
        """
        LLM-BASED GUARDRAIL - Uses your custom system prompt!
        
        This method uses the system_prompt you provided when creating the 
        GuardrailDetective instance. You can easily create different versions
        by instantiating with different prompts.
        
        Args:
            user_input: The user's message
            
        Returns:
            "ALLOW" or "BLOCK"
        """
        try:
            response_text = self._chat(
                system_prompt=self.system_prompt,
                user_message=f"Classify this input: {user_input}"
            )
            
            result = response_text.upper()
            return result if result in ["ALLOW", "BLOCK"] else "BLOCK"
            
        except Exception as e:
            print(f"LLM guardrail error: {e}")
            return "BLOCK"  # Fail-safe: block on error
    
    def _llm_batch_classify(self, user_inputs: List[str]) -> List[str]:
        """
        Process multiple inputs in a single LLM API call.
        Uses your custom system prompt for batch processing.
        
        Args:
            user_inputs: List of user messages to classify
            
        Returns:
            List of LLM classifications ("ALLOW" or "BLOCK") for each input
        """
        # Create batch version of the system prompt
        batch_system_prompt = self.system_prompt.rstrip()
        
        # Add batch-specific instructions
        if not batch_system_prompt.endswith('.'):
            batch_system_prompt += "."
        batch_system_prompt += "\n\nFor each input below, respond with only \"ALLOW\" or \"BLOCK\", one per line."
        
        # Create unified prompt with numbered inputs
        unified_prompt = "Classify each of the following inputs:\n\n"
        for i, user_input in enumerate(user_inputs, 1):
            unified_prompt += f"{i}. {user_input}\n\n"
        
        unified_prompt += f"\nRespond with exactly {len(user_inputs)} classifications, one per line (ALLOW or BLOCK):"
        
        response_text = self._chat(
            system_prompt=batch_system_prompt,
            user_message=unified_prompt
        )
        
        # Parse the response into individual classifications
        classifications = self._parse_batch_response(response_text, len(user_inputs))
        
        return classifications
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[str]:
        """
        Parse batch LLM response into individual classifications.
        
        Args:
            response_text: The raw LLM response
            expected_count: Number of classifications expected
            
        Returns:
            List of classifications, with "BLOCK" as fallback for parsing errors
        """
        lines = [line.strip().upper() for line in response_text.split('\n') if line.strip()]
        
        # Extract ALLOW classifications
        classifications = []
        for line in lines:
            # Handle numbered responses like "1. ALLOW" or just "ALLOW"
            if "ALLOW" in line:
                classifications.append("ALLOW")
            elif "BLOCK" in line:
                classifications.append("BLOCK")
        
        # Ensure we have exactly the expected number of classifications
        while len(classifications) < expected_count:
            classifications.append("BLOCK")  # Fail-safe: block if unclear
        
        return classifications[:expected_count]
    
    def custom_guardrails(self, user_input: str) -> str:
        """
        This is where you can add specific rules, keyword filtering, 
        or other custom logic that you discover through investigation.
        
        Args:
            user_input: The user's message
            
        Returns:
            "ALLOW" or "BLOCK"
        """
        text = user_input.lower()
        
        # Block obvious data extraction attempts
        if any(phrase in text for phrase in [
            "ceo phone number", "employee directory", "company secrets"
        ]):
            return "BLOCK"
        
        # Block obvious off-topic content  
        if any(phrase in text for phrase in [
            "recipe", "cooking", "sports", "politics", "weather"
        ]):
            return "BLOCK"
        
        # Add more rules here as you discover attack patterns...

        # You can uncomment the following part to use Guardrails AI to block specific patterns
        # TODO: Make sure to implement the _initialize_guard method to your liking if you uncomment the line below
        # if self.guard is not None:
        #     try:
        #         _ = self.guard.validate(user_input)
        #     except Exception as e:
        #         return "BLOCK"
        
        return "ALLOW"  # Default: allow if no custom rules triggered
    
    def _initialize_guard(self):
        """
        Initialize Guardrails AI Guard with placeholder validators.
        
        This is a placeholder implementation that you should customize based on your needs.
        
        Returns:
            Configured Guard instance or None if Guardrails AI is not available
        """
        # TODO: Following steps
        # To use this, install guardrails-ai: pip install guardrails-ai
        # Then run: guardrails configure
        # And install validators: guardrails hub install hub://guardrails/competitor_check hub://guardrails/toxic_language
        # Finally, add the correct import at the top of this script:
        # from guardrails import Guard, OnFailAction
        # from guardrails.hub import CompetitorCheck, ...

        try:
            # Example placeholder setup - customize this for your use case!
            # Note: Uncomment the lines below once you have installed guardrails-ai
            # guard = Guard().use_many(
            #     # Competitor check - blocks mentions of competitor companies
            #     CompetitorCheck(
            #         competitors=["State Farm", "Allstate", "GEICO", "Progressive", "Liberty Mutual"],
            #         on_fail=OnFailAction.EXCEPTION
            #     ),
            # )
            guard = None  # Placeholder until guardrails-ai is properly installed
            
            print("✅ Guardrails AI Guard initialized with placeholder validators")
            print("💡 TIP: Customize the _initialize_guard method to add more validators!")
            return guard
            
        except Exception as e:
            print(f"⚠️ Error initializing Guardrails AI Guard: {e}")
            print("💡 Make sure to run: guardrails configure")
            print("💡 And install validators: guardrails hub install hub://guardrails/competitor_check")
            return None
    
    def _batch_classify(self, inputs: List[str]) -> List[str]:
        """
        Process inputs in batches using the cost-efficient batch approach.
        Combines multiple inputs into single API calls to save costs.
        
        Args:
            inputs: List of user inputs to classify
            
        Returns:
            List of classifications ("ALLOW" or "BLOCK")
        """
        results = []
        
        print(f"🔍 Processing {len(inputs)} examples in batches of {self.batch_size} with cost-efficient API calls...")
        
        for i in tqdm(range(0, len(inputs), self.batch_size), desc="Classifying batches"):
            batch = inputs[i:i + self.batch_size]
            
            try:
                # Use the new batch method that makes a single API call per batch
                batch_results = self.run_batch_guardrail(batch)
                results.extend(batch_results)
            except Exception as e:
                print(f"⚠️  Error processing batch: {e}")
                # Fail-safe: block entire batch on error
                results.extend(["BLOCK"] * len(batch))
            
            # Small delay between batches to respect rate limits
            if i + self.batch_size < len(inputs):
                time.sleep(0.3)
        
        return results
    
    def evaluate_on_dataset(self, dataset_path: str) -> Dict:
        """
        Evaluate your guardrail on the mystery dataset.
        
        Args:
            dataset_path: Path to the dataset CSV file
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"📊 Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        # Get predictions
        predictions = self._batch_classify(df['input'].tolist())
        true_labels = df['expected_output'].tolist()
        
        # Calculate metrics
        metrics = self._calculate_metrics(true_labels, predictions)
        
        # Show results
        self._display_results(metrics)

        # Combine predictions and true labels into a dataframe
        df_predictions = pd.DataFrame({
            'input': df['input'], 
            'expected_output': true_labels, 
            'predicted_output': predictions,
        })

        if 'issue_type' in df.columns:
            df_predictions['issue_type'] = df['issue_type']
        
        return metrics, df_predictions
    
    def _calculate_metrics(self, true_labels: List[str], predictions: List[str]) -> Dict:
        """Calculate weighted performance metrics."""
        
        # Convert to binary for sklearn
        true_binary = [1 if label == "ALLOW" else 0 for label in true_labels]
        pred_binary = [1 if pred == "ALLOW" else 0 for pred in predictions]
        
        # Basic metrics
        accuracy = accuracy_score(true_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(true_binary, pred_binary)
        
        # Calculate raw weighted score 
        weighted_score = 0
        for true, pred in zip(true_labels, predictions):
            if true == "ALLOW" and pred == "ALLOW":  # True Positive
                weighted_score += 1
            elif true == "BLOCK" and pred == "BLOCK":  # True Negative  
                weighted_score += 1
            elif true == "ALLOW" and pred == "BLOCK":  # False Positive (blocking customers - very bad!)
                weighted_score -= 3
            elif true == "BLOCK" and pred == "ALLOW":  # False Negative (allowing malicious - bad)
                weighted_score -= 1
        
        max_possible_score = len(true_labels)
        
        return {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'weighted_score': weighted_score,
            'max_score': max_possible_score,
            'confusion_matrix': cm,
            'total_examples': len(true_labels)
        }
    
    def _display_results(self, metrics: Dict):
        """Display evaluation results."""
        
        print("\n" + "="*80)
        print("🎯 GUARDRAIL PERFORMANCE RESULTS")
        print("="*80)
        
        print(f"📈 Basic Accuracy: {metrics['accuracy']:.3f}")
        print(f"🎯 Precision: {metrics['precision']:.3f}")
        print(f"🔍 Recall: {metrics['recall']:.3f}")
        print(f"📊 F1 Score: {metrics['f1']:.3f}")
        print(f"🏆 Weighted Score: {metrics['weighted_score']}/{metrics['max_score']}")
        
        print(f"\n📋 Total Examples: {metrics['total_examples']}")
        
        # Confusion Matrix breakdown
        cm = metrics['confusion_matrix']
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"✅ True Positives (Correctly Allowed): {tp}")
            print(f"✅ True Negatives (Correctly Blocked): {tn}")
            print(f"❌ False Positives (Wrongly Blocked Legitimate): {fp} 💸")
            print(f"❌ False Negatives (Wrongly Allowed Malicious): {fn}")


    def parse_predictions_to_kaggle_format(self, predictions_df: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
        """
        Parse predictions dataframe to Kaggle submission format.
        
        Args:
            predictions_df: DataFrame containing predictions with 'predicted_output' column
            output_path: Optional path to save the Kaggle submission CSV file
            
        Returns:
            DataFrame in Kaggle format with 'id' and 'predicted_output' columns
        """
        # Create Kaggle format dataframe
        kaggle_df = pd.DataFrame({
            'id': range(0, len(predictions_df)),
            'predicted_output': predictions_df['predicted_output']  
        })
        
        # Save to CSV if output path is provided
        if output_path:
            kaggle_df.to_csv(output_path, index=False)
            print(f"✅ Kaggle submission file saved to: {output_path}")
        
        return kaggle_df

        