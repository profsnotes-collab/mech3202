import React, { useState, useEffect, useRef } from 'react';
import { BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import { auth, signOut } from './firebaseConfig'; 
import { onAuthStateChanged } from 'firebase/auth';
import styles from './index.module.css'; 
import SimpleLogin from './SimpleLogin'; 
import jokes from './jokes'; 

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [researchTopic, setResearchTopic] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentJoke, setCurrentJoke] = useState("");
  const [data, setData] = useState({
    profNotesText: "",
    images: [],
    wikiSummary: "",
    wikiImages: [],
    videos: [],
    equations: [],
    equationDescriptions: []
  });
  const [suggestions, setSuggestions] = useState([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const youtubeRef = useRef(null);
  const baseURL = window.location.origin;
  console.log('Base URL (window.location.origin):', baseURL);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setIsLoggedIn(!!user);
    });
    return unsubscribe;
  }, []);

  const handleLogout = () => {
    signOut(auth).then(() => {
      setIsLoggedIn(false);
      sessionStorage.removeItem('token');
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
      const response = await fetch(`${baseURL}/suggestions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ query: researchTopic })
      });

      if (response.ok) {
        const suggestionsData = await response.json();
        setSuggestions(suggestionsData);
      }
    } catch (error) {
      console.error("Error fetching suggestions:", error);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    setData({
      profNotesText: "",
      images: [],
      wikiSummary: "",
      wikiImages: [],
      videos: [],
      equations: [],
      equationDescriptions: []
    });

    setCurrentJoke(jokes[Math.floor(Math.random() * jokes.length)]);


    try {
      const token = sessionStorage.getItem('token');
      const response = await fetch(`${baseURL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ query: researchTopic }),
      });

      if (response.ok) {
        const data = await response.json();
        setData({
          profNotesText: data.notes_response?.Text || "",
          images: [
            { url: data.notes_response?.Image1, description: data.notes_response?.ImageDescription1 },
            { url: data.notes_response?.Image2, description: data.notes_response?.ImageDescription2 },
            { url: data.notes_response?.Image3, description: data.notes_response?.ImageDescription3 }
          ].filter(img => img.url),
          wikiSummary: data.wikipedia_response?.Summary?.trim() || 'No summary available',
          wikiImages: Object.values(data.wikipedia_response?.["Top Three Images"] || []),
          videos: data.youtube_response?.results || [],
          equations: [data.notes_response?.Equation1, data.notes_response?.Equation2, data.notes_response?.Equation3].filter(eq => eq),
          equationDescriptions: [data.notes_response?.EquationDescription1, data.notes_response?.EquationDescription2, data.notes_response?.EquationDescription3].filter(desc => desc)
        });
      }
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    fetchData();
  };

  const highlightMatch = (text, query) => {
    const startIndex = text.toLowerCase().indexOf(query.toLowerCase());
    if (startIndex === -1) return text;
    const endIndex = startIndex + query.length;
    return (
      <>
        {text.slice(0, startIndex)}
        <strong>{text.slice(startIndex, endIndex)}</strong>
        {text.slice(endIndex)}
      </>
    );
  };

  const displaySuggestions = (suggestions) => {
    return suggestions.map((suggestion, index) => (
      <div key={index} className={styles.suggestionItem} onClick={() => setResearchTopic(suggestion)}>
        {highlightMatch(suggestion, researchTopic)}
      </div>
    ));
  };

  if (!isLoggedIn) {
    return <SimpleLogin onLogin={setIsLoggedIn} />;
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.menuToggle} onClick={() => setMenuOpen(!menuOpen)}>Menu</div>
        <div className={styles.title}>Professor Notes AI Assistant</div>
        <div className={styles.userSection} onClick={() => setUserMenuOpen(!userMenuOpen)}>
          User
          {userMenuOpen && (
            <div className={styles.userMenu}>
              <button className={styles.logoutButton} onClick={handleLogout}>Logout</button>
            </div>
          )}
        </div>
      </header>

      {menuOpen && (
        <nav className={styles.menu}>
          <button onClick={() => youtubeRef.current.scrollIntoView({ behavior: 'smooth' })}>YouTube Video</button>
          <button>Settings</button>
          <button className={styles.logoutButton} onClick={handleLogout}>Logout</button>
        </nav>
      )}

      <form className={styles.searchForm} onSubmit={handleSearch}>
        <input
          type="text"
          value={researchTopic}
          onChange={(e) => setResearchTopic(e.target.value)}
          onKeyUp={fetchSuggestions}
          placeholder="Enter your research topic"
        />
        <button type="submit">Search</button>
        {suggestions.length > 0 && (
          <div className={styles.suggestions}>
            {displaySuggestions(suggestions)}
          </div>
        )}
      </form>

      {loading ? (
        <div className={styles.loaderContainer}>
          <div className={styles.loader}></div>
          <p className={styles.joke}>{currentJoke}</p>
        </div>
      ) : (
        <main className={styles.content}>
          <section className={styles.section}>
            <h2>Prof Notes</h2>
            <p>{data.profNotesText}</p>
            {data.images.map((img, index) => (
              <div key={index}>
                <img src={img.url} alt={`Image ${index}`} className={styles.image} />
                <p>{img.description}</p>
              </div>
            ))}
            {data.equations.map((eq, index) => (
              <div key={index}>
                <BlockMath math={eq} />
                <p>{data.equationDescriptions[index]}</p>
              </div>
            ))}
          </section>

          <section className={styles.section}>
            <h2>Wikipedia Summary</h2>
            <p>{data.wikiSummary}</p>
            {data.wikiImages.map((img, index) => (
              <div key={index}>
                <img src={img.URL} alt={`Wiki Image ${index}`} className={styles.image} />
                <p>{img.Description}</p>
              </div>
            ))}
          </section>

          <section className={`${styles.section} ${styles.youtubeWindow}`} ref={youtubeRef}>
            <h2>YouTube Videos</h2>
            {data.videos.length > 0 ? (
              data.videos.map((video, index) => (
                <div key={index}>
                  <iframe
                    src={`https://www.youtube.com/embed/${video.video_url.split('v=')[1]}`}
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                    title={`YouTube Video ${index}`}
                  ></iframe>
                  <div>{video.title}</div>
                  <div>{video.description}</div>
                  <a href={video.timestamp} target="_blank" rel="noopener noreferrer">Watch at timestamp</a>
                </div>
              ))
            ) : (
              <p>No YouTube videos available for this topic.</p>
            )}
          </section>
        </main>
      )}

      <footer className={styles.footer}>
        <p>© ARIA 2024</p>
      </footer>
    </div>
  );
}

export default App;