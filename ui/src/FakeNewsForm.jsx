import React, { useState } from 'react';
import axios from 'axios';
import './FakeNewsForm.css';

const FakeNewsForm = () => {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [confidence, setConfidence] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setResult(null);
        setConfidence(null);

        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', { text });
            // Extract result and confidence from response
            const { result: prediction, confidence: score } = response.data;

            // Update state with the response data
            setResult(prediction);
            setConfidence(score);
        } catch (error) {
            console.error("Error making prediction:", error);
            setResult("Error making prediction. Please try again.");
        }

        setLoading(false);
    };

    return (
        <div className="page-container">
            <div className="container">
                <h2 className="title">Fake News Detection</h2>
                <form onSubmit={handleSubmit} className="form">
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Enter news article text"
                        rows="10"
                        className="textarea"
                    ></textarea>
                    <button type="submit" className="button">Check News</button>
                </form>
                {loading && <div className="spinner"></div>}
                {result && !loading && (
                    <h3 className="result">
                        Prediction: {result} <br />
                        Confidence: {(confidence * 100).toFixed(2)}%
                    </h3>
                )}
            </div>
        </div>
    );
};

export default FakeNewsForm;
