import React, { useState, useEffect, useRef } from 'react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { auth } from './firebaseConfig';
import { signOut, onAuthStateChanged } from 'firebase/auth';
import styles from './index.module.css';
import innovation from './assets/innovation.png';
import menuIcon from './assets/menu.png';
import userIcon from './assets/user.png';
import SimpleLogin from './SimpleLogin';
import jokes from './jokes';

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [researchTopic, setResearchTopic] = useState("");
  const [wikiSummary, setWikiSummary] = useState("");
  const [profNotesText, setProfNotesText] = useState("");
  const [images, setImages] = useState([]);
  const [wikiImages, setWikiImages] = useState([]);
  const [videos, setVideos] = useState([]);
  const [equations, setEquations] = useState([]);
  const [equationDescriptions, setEquationDescriptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const youtubeRef = useRef(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [currentJoke, setCurrentJoke] = useState("");
  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setIsLoggedIn(!!user);
    });

    return unsubscribe;
  }, []);

  const handleLogout = () => {
    signOut(auth).then(() => {
      setIsLoggedIn(false);
      sessionStorage.removeItem('token'); // Clear token on logout
    }).catch((error) => {
      console.error("Error signing out:", error);
    });
  };

  const fetchSuggestions = async () => {
    if (researchTopic.length < 2) {
      setSuggestions([]);
      return;
    }

    try {
      const token = sessionStorage.getItem('token');
      const response = await fetch('https://api-gateway-dot-profsnotes.appspot.com/suggestions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ query: researchTopic })
      });

      const suggestionsData = await response.json();
      console.log('Suggestions:', suggestionsData);
      setSuggestions(suggestionsData);
    } catch (error) {
      console.error("Error fetching suggestions:", error);
    }
  };

  const fetchEquationsAndText = async () => {
    try {
      const token = sessionStorage.getItem('token');
      const response = await fetch("https://api-gateway-dot-profsnotes.appspot.com/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ query: researchTopic }),
      });
      const data = await response.json();
      setProfNotesText(data.notes_response?.Text || "");
      setImages([
        { url: data.notes_response?.Image1, description: data.notes_response?.ImageDescription1 },
        { url: data.notes_response?.Image2, description: data.notes_response?.ImageDescription2 },
        { url: data.notes_response?.Image3, description: data.notes_response?.ImageDescription3 },
      ].filter(img => img.url));
      setEquations([data.notes_response?.Equation1, data.notes_response?.Equation2, data.notes_response?.Equation3].filter(eq => eq));
      setEquationDescriptions([data.notes_response?.EquationDescription1, data.notes_response?.EquationDescription2, data.notes_response?.EquationDescription3].filter(desc => desc));
    } catch (error) {
      console.error("Error fetching equations and text:", error);
    }
  };

  const fetchWikiSummary = async () => {
    try {
      const token = sessionStorage.getItem('token');
      const response = await fetch("https://api-gateway-dot-profsnotes.appspot.com/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ query: researchTopic }),
      });
      const data = await response.json();
      if (data.wikipedia_response?.error) {
        setWikiSummary('No summary available');
        setWikiImages([]);
      } else {
        setWikiSummary(data.wikipedia_response?.Summary?.trim() || 'No summary available');
        setWikiImages(Object.values(data.wikipedia_response?.["Top Three Images"] || []));
      }
    } catch (error) {
      console.error("Error fetching Wikipedia summary:", error);
    }
  };

  const fetchYouTubeVideo = async () => {
    try {
      const token = sessionStorage.getItem('token');
      const response = await fetch("https://api-gateway-dot-profsnotes.appspot.com/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ query: researchTopic }),
      });
      const data = await response.json();
      console.log("YouTube API response:", data);
      setVideos(data.youtube_response?.results || []);
    } catch (error) {
      console.error("Error fetching YouTube videos:", error);
    }
  };

  const handleSearch = async () => {
    setWikiSummary("");
    setProfNotesText("");
    setImages([]);
    setWikiImages([]);
    setEquations([]);
    setEquationDescriptions([]);
    setVideos([]);
    setLoading(true);
    setCurrentJoke(jokes[Math.floor(Math.random() * jokes.length)]);
    const promises = [fetchEquationsAndText(), fetchWikiSummary(), fetchYouTubeVideo()];
    await Promise.all(promises);
    setLoading(false);
    if (youtubeRef.current) {
      youtubeRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToYoutube = () => {
    if (youtubeRef.current) {
      youtubeRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const toggleUserMenu = () => {
    setUserMenuOpen(!userMenuOpen);
  };

  const sanitizeLatex = (latex) => {
    return latex && latex.trim() ? latex : "\\text{Invalid LaTeX}";
  };

  const renderEquation = (latex) => {
    try {
      return <BlockMath math={latex} />;
    } catch (error) {
      console.error("Error rendering LaTeX:", error);
      return <span className={styles.invalidLatex}>Invalid LaTeX</span>;
    }
  };

  if (!isLoggedIn) {
    return <SimpleLogin onLogin={setIsLoggedIn} />;
  }

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
        <div className={styles.userSection} onClick={toggleUserMenu}>
          <img src={userIcon} alt="User" className={styles.userIcon} />
          <span className={styles.userName}>User</span>
          {userMenuOpen && (
            <div className={styles.userMenu}>
              <button onClick={handleLogout} className={styles.logoutButton}>Logout</button>
            </div>
          )}
        </div>
      </div>
      {menuOpen && (
        <div className={styles.menu}>
          <button onClick={scrollToYoutube}>YouTube Video</button>
          <button>Settings</button>
          <button onClick={handleLogout} className={styles.logoutButton}>Logout</button>
        </div>
      )}

      <form className={styles.form} onSubmit={(e) => { e.preventDefault(); handleSearch(); }}>
        <input
          type="text"
          name="research-topic"
          placeholder="Enter your research topic"
          onChange={(e) => {
            setResearchTopic(e.target.value);
            fetchSuggestions();
          }}
          className={styles.input}
          value={researchTopic}
          autoComplete="off"
        />
        <button type="submit" className={styles.searchButton}>Search</button>
        {suggestions.length > 0 && (
          <div className={styles.suggestions}>
            {suggestions.map((suggestion, index) => (
              <div
                key={index}
                className={styles.suggestionItem}
                onClick={() => {
                  setResearchTopic(suggestion);
                  setSuggestions([]); // Hide suggestions after selecting one
                  handleSearch();
                }}
              >
                {suggestion}
              </div>
            ))}
          </div>
        )}
      </form>

      {loading && (
        <div className={styles.loaderContainer}>
          <div className={styles.loader}></div>
          <p className={styles.joke}>{currentJoke}</p>
        </div>
      )}

      {!loading && (
        <div className={styles.content}>
          <div className={`${styles.window} ${styles.hoverEffect}`}>
            <h4>Prof Notes Text:</h4>
            <p>{profNotesText}</p>
            {equations.length > 0 ? (
              equations.map((eq, index) => (
                <React.Fragment key={index}>
                  {renderEquation(sanitizeLatex(eq))}
                  <p>{equationDescriptions[index]}</p>
                </React.Fragment>
              ))
            ) : (
              images.map((img, index) => (
                <div key={index}>
                  <img src={img.url} alt={`Prof Notes Image ${index}`} className={styles.image} />
                  <p>{img.description}</p>
                </div>
              ))
            )}
          </div>
          <div className={`${styles.window} ${styles.hoverEffect}`}>
            <h4>Wikipedia Summary:</h4>
            <p>{wikiSummary}</p>
            {wikiImages.map((img, index) => (
              <div key={index}>
                <img src={img.URL} alt={`Wiki Image ${index}`} className={styles.image} />
                <p>{img.Description}</p>
                <p>{img.Caption}</p>
              </div>
            ))}
          </div>
          <div className={`${styles.window} ${styles.hoverEffect} ${styles.youtubeWindow}`} ref={youtubeRef}>
            <h4>YouTube Videos:</h4>
            {videos.length > 0 ? (
              videos.map((video, index) => (
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
              ))
            ) : (
              <p>No YouTube videos available for this topic.</p>
            )}
          </div>
        </div>
      )}

      <footer className={styles.footer}>
        <p>© ARIA 2024</p>
      </footer>
    </main>
  );
}

export default App;
