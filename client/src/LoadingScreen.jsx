import React, { useState, useEffect } from 'react';
import styles from './LoadingScreen.module.css'; 

const LoadingScreen = ({ messages = ["Doing important stuff...", "Just a couple more seconds..."], interval = 3000, className = '' }) => {
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentMessageIndex((prevIndex) => (prevIndex + 1) % messages.length);
    }, interval);

    return () => clearInterval(timer);
  }, [messages, interval]);

  return (
    <div className={`${styles.loadingScreen} ${className}`}>
      <div className={styles.loadingWheel}></div>
      <p className={styles.loadingMessage}>{messages[currentMessageIndex]}</p>
    </div>
  );
};

export default LoadingScreen;