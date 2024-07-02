import React, { useState, useRef } from 'react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { db } from './firebaseConfig';
import { collection, addDoc } from 'firebase/firestore';
import styles from './index.module.css';
import innovation from './assets/innovation.png';
import menuIcon from './assets/menu.png';
import userIcon from './assets/user.png';

function App() {
  const [researchTopic, setResearchTopic] = useState("");
  const [wikiSummary, setWikiSummary] = useState("");
  const [profNotesText, setProfNotesText] = useState("");
  const [images, setImages] = useState([]);
  const [videos, setVideos] = useState([]);
  const [equation1, setEquation1] = useState("");
  const [equation2, setEquation2] = useState("");
  const [equationDescription1, setEquationDescription1] = useState("");
  const [equationDescription2, setEquationDescription2] = useState("");
  const [loading, setLoading] = useState(false);
  const youtubeRef = useRef(null);
  const [menuOpen, setMenuOpen] = useState(false);

  const saveSearchToFirestore = async (query) => {
    try {
      await addDoc(collection(db, "User-Queries/Queries"), {
        query,
        timestamp: new Date()
      });
      console.log("Search saved to Firestore");
    } catch (e) {
      console.error("Error adding document: ", e);
    }
  };

  const fetchEquationsAndText = async () => {
    setLoading(true);
    try {
      const response = await fetch("https://notes-service-dot-profsnotes.appspot.com/query_technical", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: researchTopic }),
      });
      const data = await response.json();
      setEquation1(data.Equation1);
      setEquation2(data.Equation2);
      setEquationDescription1(data.EquationDescription1);
      setEquationDescription2(data.EquationDescription2);
      setProfNotesText(data.Text);
      saveSearchToFirestore(researchTopic);
    } catch (error) {
      console.error("Error fetching equations and text:", error);
    }
    setLoading(false);
  };

  const fetchWikiSummary = async (topic) => {
    setLoading(true);
    try {
      const response = await fetch("https://wikipedia-service-dot-profsnotes.appspot.com/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: topic }),
      });

      const data = await response.json();
      if (data.error) {
        setWikiSummary('No summary available');
        setImages([]);
      } else {
        setWikiSummary(data.Summary ? data.Summary.trim() : 'No summary available');
        setImages(Object.values(data["Top Three Images"]) || []);
      }
    } catch (error) {
      console.error("Error fetching Wikipedia summary:", error);
    }
    setLoading(false);
  };

  const fetchYouTubeVideo = async (topic) => {
    setLoading(true);
    try {
      const response = await fetch("https://youtube-service-dot-profsnotes.appspot.com/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: topic }),
      });

      const data = await response.json();
      setVideos(data);
    } catch (error) {
      console.error("Error fetching YouTube videos:", error);
    }
    setLoading(false);
  };

  const handleWikiSummary = async () => {
    await fetchWikiSummary(researchTopic);
  };

  const handleYouTubeVideo = async () => {
    await fetchYouTubeVideo(researchTopic);
    youtubeRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  const handleProfNotes = async () => {
    await fetchEquationsAndText();
  };

  const scrollToYoutube = () => {
    youtubeRef.current.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  return (
    <main className={styles.main}>
      <div className={styles.header}>
        <div className={styles.iconContainer} onClick={toggleMenu}>
          <img src={menuIcon} alt="Menu" className={styles.menuIcon} />
        </div>
        <div className={styles.titleContainer}>
          <img src={innovation} alt="App Logo" className={styles.appLogo} />
          <h3>Professor Notes AI Assistant</h3>
        </div>
        <div className={styles.userSection}>
          <img src={userIcon} alt="User" className={styles.userIcon} />
          <span className={styles.userName}>User</span>
        </div>
      </div>
      {menuOpen && (
        <div className={styles.menu}>
          <button onClick={scrollToYoutube}>YouTube Video</button>
          <button>Settings</button>
          <button>Logout</button>
        </div>
      )}

      <form className={styles.form}>
        <input
          type="text"
          name="research-topic"
          placeholder="Enter your research topic"
          onChange={(e) => setResearchTopic(e.target.value)}
          className={styles.input}
        />
        <button type="button" className={styles.submitButton} onClick={handleProfNotes}>Search Prof Notes</button>
        <button type="button" className={styles.submitButton} onClick={handleWikiSummary}>Wikipedia Search</button>
      </form>

      {loading && <div className={styles.loader}></div>}

      <div className={styles.content}>
        <div className={`${styles.window} ${styles.hoverEffect}`}>
          <h4>Prof Notes Text:</h4>
          <p>{profNotesText}</p>
          <h4>Equation 1:</h4>
          <BlockMath math={equation1} />
          <p>{equationDescription1}</p>
          <h4>Equation 2:</h4>
          <BlockMath math={equation2} />
          <p>{equationDescription2}</p>
        </div>
        <div className={`${styles.window} ${styles.hoverEffect}`}>
          <h4>Wikipedia Summary:</h4>
          <p>{wikiSummary}</p>
          {images.map((img, index) => (
            <div key={index}>
              <img src={img.URL} alt={`Wiki Image ${index}`} className={styles.image} />
              <p>{img.Description}</p>
              <p>{img.Caption}</p>
            </div>
          ))}
        </div>
        <div className={`${styles.window} ${styles.hoverEffect} ${styles.youtubeWindow}`}>
          <h4>YouTube Videos:</h4>
          {videos.map((video, index) => (
            <div key={index} className={styles.youtubeContainer}>
              <iframe
                src={`https://www.youtube.com/embed/${video.video_url.split('v=')[1]}`}
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                title={`YouTube Video ${index}`}
              ></iframe>
              <div className={styles.youtubeVideoTitle}>{video.title}</div>
              <div className={styles.youtubeVideoDescription}>{video.description}</div>
              <div className={styles.youtubeVideoText}>{video.text}</div>
              <a href={video.timestamp} target="_blank" rel="noopener noreferrer" className={styles.youtubeTimestamp}>Watch at timestamp</a>
            </div>
          ))}
          <button className={styles.button} onClick={handleYouTubeVideo}>Watch More YouTube Videos</button>
        </div>
      </div>

      <footer className={styles.footer}>
        <p>© ARIA 2024</p>
      </footer>
    </main>
  );
}

export default App;
