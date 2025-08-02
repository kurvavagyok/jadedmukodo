
// --- Globális Állapot és Konfiguráció ---
const AppState = {
    isTyping: false,
    userId: 'user_' + Math.random().toString(36).substr(2, 9),
    isDeepSearchActive: false,
};
const API_BASE_URL = '/api';

// --- DOM Elemek Gyors Elérése ---
const getEl = (id) => document.getElementById(id);
let sidebar, sidebarOverlay, sidebarContent, messagesContainer, messageInput, sendBtn, welcomeScreen;

// --- Szinkronizált Téma Kezelő ---
class DynamicThemeManager {
    constructor() {
        this.root = document.documentElement;
        this.auroraBlobs = document.querySelectorAll('.aurora-background .aurora');

        // Csak az engedélyezett két-szín kombinációk (TILOS: piros, narancs, barna, szürke)
        this.colorPairs = [
            { primary: '#1565C0', secondary: '#39FF14' }, // sötétkék-neonfzöld
            { primary: '#007AFF', secondary: '#FF2D55' }, // kék-pink
            { primary: '#FF2D55', secondary: '#AF52DE' }, // pink-lila
            { primary: '#AF52DE', secondary: '#007AFF' }, // lila-kék
            { primary: '#007AFF', secondary: '#00E676' }, // kék-zöld
            { primary: '#00E676', secondary: '#FFFF00' }  // zöld-sárga
        ];
        this.currentIndex = 0;

        this.updateColors(this.colorPairs[0]);
        setInterval(() => this.cyclePalette(), 90000);

        setTimeout(() => {
            this.auroraBlobs.forEach(b => b.classList.add('visible'));
        }, 500);
    }

    updateColors = (colorPair) => {
        const primaryRGB = this.hexToRgb(colorPair.primary);
        const secondaryRGB = this.hexToRgb(colorPair.secondary);

        // Kétszínű gradientek használata
        this.root.style.setProperty('--primary', colorPair.primary);
        this.root.style.setProperty('--secondary', colorPair.secondary);
        this.root.style.setProperty('--accent', colorPair.primary);
        this.root.style.setProperty('--border-glow', `rgba(${primaryRGB}, 0.5)`);
        this.root.style.setProperty('--primary-rgb', primaryRGB);
        this.root.style.setProperty('--secondary-rgb', secondaryRGB);

        // Aurora háttér két színnel
        this.auroraBlobs.forEach((blob, index) => {
            blob.style.backgroundColor = index % 2 === 0 ? colorPair.primary : colorPair.secondary;
        });
    };

    hexToRgb = (hex) => {
        let r = 0, g = 0, b = 0;
        if (hex.length === 7) {
            r = parseInt(hex.substring(1, 3), 16);
            g = parseInt(hex.substring(3, 5), 16);
            b = parseInt(hex.substring(5, 7), 16);
        }
        return `${r}, ${g}, ${b}`;
    };

    cyclePalette = () => {
        this.currentIndex = (this.currentIndex + 1) % this.colorPairs.length;
        const nextColorPair = this.colorPairs[this.currentIndex];
        this.updateColors(nextColorPair);
    };
}

// --- API Kommunikáció ---
async function fetchAPI(endpoint, options = {}) {
    try {
        const url = endpoint.startsWith('/api') ? endpoint : `${API_BASE_URL}${endpoint}`;
        const response = await fetch(url, {
            ...options,
            headers: { 'Content-Type': 'application/json', ...options.headers }
        });
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        hideTypingIndicator();
        addAiMessage(`Hiba történt a szerverrel való kommunikáció során: ${error.message}`);
        throw error;
    }
}

// --- ÚJ SVG LOGÓ ---
function createSVGLogo() {
    const container = getEl('logo-icon-container');
    if(container) {
        container.innerHTML = `
            <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: 100%;">
                <defs>
                    <linearGradient id="logo-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="var(--primary)" />
                        <stop offset="100%" stop-color="var(--secondary)" />
                    </linearGradient>
                </defs>
                <path d="M 30 28 C 30 50 50 72 70 72" stroke="url(#logo-gradient)" stroke-width="8" fill="none" stroke-linecap="round"/>
                <circle cx="30" cy="72" r="6" fill="url(#logo-gradient)"/>
            </svg>
        `;
    }
}

// --- EGYSZERŰ SIDEBAR GESTURE KEZELŐ ---
class SimpleSidebarGesture {
    constructor() {
        this.setupSidebarGesture();
    }

    setupSidebarGesture() {
        // Csak sidebar swipe gesture
        document.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        document.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        document.addEventListener('touchend', this.handleTouchEnd.bind(this), { passive: false });
    }

    handleTouchStart(e) {
        this.startX = e.touches[0].clientX;
        this.startY = e.touches[0].clientY;
    }

    handleTouchMove(e) {
        if (!this.startX || !this.startY) return;

        const currentX = e.touches[0].clientX;
        const currentY = e.touches[0].clientY;
        const deltaX = currentX - this.startX;
        const deltaY = currentY - this.startY;

        // Ha vertikális görgetés történik a chat területen, ne zavarjuk meg
        const isInChatArea = e.target.closest('.messages-container');
        if (isInChatArea && Math.abs(deltaY) > Math.abs(deltaX)) {
            return; // Engedjük a vertikális görgetést
        }

        // Sidebar swipe detection
        if (Math.abs(deltaX) > 50 && Math.abs(deltaX) > Math.abs(deltaY)) {
            if (deltaX > 0 && this.startX < 50) {
                // Swipe right from left edge - open sidebar
                openSidebar();
                this.startX = null;
                this.startY = null;
            } else if (deltaX < -50 && sidebar.classList.contains('open')) {
                // Swipe left when sidebar is open - close sidebar
                closeSidebar();
                this.startX = null;
                this.startY = null;
            }
        }
    }

    handleTouchEnd(e) {
        this.startX = null;
        this.startY = null;
    }
}

// Segédfüggvények a gesture-ökhöz
function scrollToTop() {
    messagesContainer.scrollTo({ top: 0, behavior: 'smooth' });
}

function cyclePalette() {
    if (window.themeManager) {
        window.themeManager.cyclePalette();
    }
}

function toggleTheme() {
    // Téma váltás implementáció
    document.body.classList.toggle('dark-theme');
}

// --- Alkalmazás Inicializálása Gesture-ökkel ---
document.addEventListener('DOMContentLoaded', () => {
    try {
        // DOM elemek inicializálása
        sidebar = getEl('sidebar');
        sidebarOverlay = getEl('sidebarOverlay');
        sidebarContent = getEl('sidebarContent');
        messagesContainer = getEl('messagesContainer');
        messageInput = getEl('messageInput');
        sendBtn = getEl('sendBtn');
        welcomeScreen = getEl('welcomeScreen');

        window.themeManager = new DynamicThemeManager();
        createSVGLogo();
        setupEventListeners();
        loadServices();

        // Sidebar gesture
        window.gestureManager = new SimpleSidebarGesture();

        // Focus input ha elérhető
        if (messageInput) {
            messageInput.focus();
        }

        console.log('JADED alkalmazás sikeresen inicializálva');
    } catch (error) {
        console.error('Inicializálási hiba:', error);
    }
});

function setupEventListeners() {
    if (messageInput) {
        messageInput.addEventListener('input', () => {
            if (messageInput.value.trim().length > 0) {
                sendBtn.classList.add('active');
            } else {
                sendBtn.classList.remove('active');
            }
            messageInput.style.height = 'auto';
            messageInput.style.height = `${Math.min(messageInput.scrollHeight, 120)}px`;
        });

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!sendBtn.disabled) sendMessage();
            }
        });
    }

    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', closeSidebar);
    }

    // Menu toggle eseménykezelő
    const menuToggle = document.querySelector('.menu-toggle');
    if (menuToggle) {
        menuToggle.addEventListener('click', toggleSidebar);
    }

    // Send button eseménykezelő
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    // Deep search toggle eseménykezelő
    const deepSearchToggle = getEl('deepSearchToggle');
    if (deepSearchToggle) {
        deepSearchToggle.addEventListener('click', toggleDeepSearch);
    }

    // New chat button eseménykezelő
    const newChatBtn = document.querySelector('.new-chat-btn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', newChat);
    }

    // Start button eseménykezelő
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('start-btn')) {
            hideWelcome();
        }
    });
}

// --- UI Kezelő Függvények ---
function openSidebar() { 
    if (sidebar) sidebar.classList.add('open'); 
    if (sidebarOverlay) sidebarOverlay.classList.add('active'); 
}

function closeSidebar() { 
    if (sidebar) sidebar.classList.remove('open'); 
    if (sidebarOverlay) sidebarOverlay.classList.remove('active'); 
}

// Sidebar toggle funkció
function toggleSidebar() {
    if (sidebar && sidebar.classList.contains('open')) {
        closeSidebar();
    } else {
        openSidebar();
    }
}

// Globálisan elérhető függvények
window.toggleSidebar = toggleSidebar;
window.openSidebar = openSidebar;
window.closeSidebar = closeSidebar;
window.newChat = newChat;
window.hideWelcome = hideWelcome;
window.selectService = selectService;
window.toggleCategory = toggleCategory;
window.sendMessage = sendMessage;
window.toggleDeepSearch = toggleDeepSearch;

function hideWelcome() { 
    if (welcomeScreen) { 
        welcomeScreen.style.display = 'none'; 
        welcomeScreen = null; 
    } 
}

function newChat() {
    messagesContainer.innerHTML = '';
    const newWelcomeScreen = document.createElement('header');
    newWelcomeScreen.className = 'welcome-screen';
    newWelcomeScreen.id = 'welcomeScreen';
    newWelcomeScreen.innerHTML = `
        <h1 class="welcome-title">JADED</h1>
        <p class="welcome-subtitle">Készen állok a következő feladatra. Válassz egy szolgáltatást, vagy kérdezz bármit!</p>
        <button class="start-btn" onclick="hideWelcome()"><i class="fas fa-rocket"></i> Kezdjük el!</button>
    `;
    messagesContainer.appendChild(newWelcomeScreen);
    welcomeScreen = getEl('welcomeScreen');
    messageInput.value = '';
    messageInput.dispatchEvent(new Event('input'));
    messageInput.focus();
    closeSidebar();
}

// --- Szolgáltatások Betöltése ---
async function loadServices() {
    try {
        const data = await fetchAPI('/services');
        if (!data || !data.categories) {
            throw new Error("Invalid data structure from /services");
        }
        const { categories } = data;
        sidebarContent.innerHTML = '';
        const categoryMeta = {
            biologiai_orvosi: { name: 'Biológiai & Orvosi', icon: 'fa-dna' },
            kemiai_anyagtudomanyi: { name: 'Kémiai & Anyagtudományi', icon: 'fa-atom' },
            kornyezeti_fenntarthato: { name: 'Környezeti & Fenntartható', icon: 'fa-leaf' },
            fizikai_asztrofizikai: { name: 'Fizikai & Asztrofizikai', icon: 'fa-satellite-dish' },
            technologiai_melymu: { name: 'Technológiai & Mélyműszaki', icon: 'fa-microchip' },
            tarsadalmi_gazdasagi: { name: 'Társadalmi & Gazdasági', icon: 'fa-chart-line' }
        };

        for (const categoryId in categories) {
            const category = categories[categoryId];
            const meta = categoryMeta[categoryId] || { name: categoryId, icon: 'fa-star' };
            let servicesHTML = '';
            for (const serviceName in category) {
                const safeServiceName = serviceName.replace(/'/g, '\\\'');
                const safeDescription = category[serviceName].replace(/'/g, '\\\'');
                servicesHTML += `
                    <div class="service-item" onclick="selectService('${safeServiceName}', '${safeDescription}')">
                        <div class="service-icon"><i class="fas fa-flask"></i></div>
                        <div class="service-info">
                            <h4>${escapeHtml(serviceName)}</h4>
                            <p>${escapeHtml(category[serviceName])}</p>
                        </div>
                    </div>`;
            }
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'service-category';
            categoryDiv.innerHTML = `
                <div class="category-header" onclick="toggleCategory(this)">
                    <div class="category-title"><i class="fas ${meta.icon}"></i><span>${meta.name}</span></div>
                    <i class="fas fa-chevron-down"></i>
                </div>
                <div class="category-services"><div>${servicesHTML}</div></div>`;
            sidebarContent.appendChild(categoryDiv);
        }
    } catch (error) {
        sidebarContent.innerHTML = '<p style="padding: 1rem; color: var(--text-muted);">Szolgáltatások betöltése sikertelen.</p>';
    }
}

function toggleCategory(headerElement) {
    const services = headerElement.nextElementSibling;
    const chevron = headerElement.querySelector('.fa-chevron-down');
    const isOpen = services.classList.toggle('open');
    chevron.style.transform = isOpen ? 'rotate(180deg)' : 'rotate(0deg)';
}

// --- Deep Search Toggle Functionality ---
function toggleDeepSearch() {
    const toggle = getEl('deepSearchToggle');
    AppState.isDeepSearchActive = !AppState.isDeepSearchActive;

    if (AppState.isDeepSearchActive) {
        toggle.classList.add('active');
        messageInput.placeholder = 'Deep Search kutatási téma...';
    } else {
        toggle.classList.remove('active');
        messageInput.placeholder = 'Írj egy üzenetet...';
    }
}

// --- Üzenetküldés és Megjelenítés ---
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || AppState.isTyping) return;

    hideWelcome();

    if (AppState.isDeepSearchActive) {
        // Deep Search indítása
        addUserMessage(`Deep Research indítva: ${message}`);
        messageInput.value = '';
        messageInput.dispatchEvent(new Event('input'));
        messageInput.style.height = 'auto';
        showResearchIndicator(message);

        // Deep Search deaktiválása küldés után
        AppState.isDeepSearchActive = false;
        getEl('deepSearchToggle').classList.remove('active');
        messageInput.placeholder = 'Írj egy üzenetet...';

        try {
            const data = await fetchAPI('/deep_discovery/deep_research', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: message, user_id: AppState.userId })
            });

            AppState.researchState = {
                query: data.query,
                documentContent: data.final_synthesis,
                sources: data.sources || [],
                totalSources: data.total_sources_found || 0
            };

            getEl('researchIndicator')?.remove();
            displayResearchCompletion();
        } catch (error) {
            getEl('researchIndicator')?.remove();
        }
    } else {
        // Normál chat üzenet
        addUserMessage(message);
        messageInput.value = '';
        messageInput.dispatchEvent(new Event('input'));
        messageInput.style.height = 'auto';
        showTypingIndicator();
        try {
            const data = await fetchAPI('/deep_discovery/chat', {
                method: 'POST',
                body: JSON.stringify({ message: message, user_id: AppState.userId })
            });
            hideTypingIndicator();
            addAiMessage(data.response);
        } catch (error) {
            // Az error kezelése már a fetchAPI-ban megtörténik
        }
    }
}

function addUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    messageDiv.innerHTML = `<div class="user-bubble">${escapeHtml(message)}</div>`;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function addAiMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ai-message';
    const parsedContent = marked.parse(content);
    messageDiv.innerHTML = `<div class="ai-content">${parsedContent}</div>`;
    messagesContainer.appendChild(messageDiv);
    messageDiv.querySelectorAll('pre code').forEach(hljs.highlightElement);
    scrollToBottom();
}

function showTypingIndicator() {
    if (AppState.isTyping) return;
    AppState.isTyping = true;
    sendBtn.disabled = true;
    const indicator = document.createElement('div');
    indicator.id = 'typingIndicator';
    indicator.className = 'message ai-message';
    indicator.innerHTML = `
        <div class="ai-content" style="padding: 16px 18px; width: min-content;">
            <div class="typing-indicator">
                <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
            </div>
        </div>`;
    messagesContainer.appendChild(indicator);
    scrollToBottom();
}

function hideTypingIndicator() {
    AppState.isTyping = false;
    sendBtn.disabled = false;
    getEl('typingIndicator')?.remove();
}

function scrollToBottom() {
    if (messagesContainer) {
        // Többféle módszer a biztos görgetéshez
        try {
            messagesContainer.scrollTo({ 
                top: messagesContainer.scrollHeight, 
                behavior: 'smooth' 
            });
        } catch (e) {
            // Fallback régi böngészőknek
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // iOS Safari fix
        setTimeout(() => {
            if (messagesContainer.scrollTop < messagesContainer.scrollHeight - messagesContainer.clientHeight) {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }, 100);
    }
}

// --- Segédfüggvények ---
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

// Debug függvény
function debugLog(message, data = null) {
    console.log(`[JADED Debug] ${message}`, data);
}

// Szolgáltatás kiválasztás függvény
function selectService(serviceName, serviceDescription) {
    hideWelcome();
    const prompt = `Kérlek segíts a következő szolgáltatással: ${serviceName} - ${serviceDescription}`;
    if (messageInput) {
        messageInput.value = prompt;
        messageInput.dispatchEvent(new Event('input'));
        messageInput.focus();
    }
    closeSidebar();
}

// Kategória toggle függvény
function toggleCategory(headerElement) {
    if (!headerElement) return;
    
    const services = headerElement.nextElementSibling;
    const chevron = headerElement.querySelector('.fa-chevron-down');
    
    if (services && chevron) {
        const isOpen = services.classList.toggle('open');
        chevron.style.transform = isOpen ? 'rotate(180deg)' : 'rotate(0deg)';
    }
}

// --- Deep Search Progress Indicator ---
function showResearchIndicator(query) {
    const indicatorDiv = document.createElement('div');
    indicatorDiv.className = 'message ai-message';
    indicatorDiv.id = 'researchIndicator';
    indicatorDiv.innerHTML = `
        <div class="research-indicator">
            <div class="research-header">
                <div class="spinner"></div>
                <h4>Háromszoros AI Keresési Rendszer</h4>
            </div>
            <div class="progress-section">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill" style="width: 0%"></div>
                </div>
                <div class="progress-text">
                    <span id="progressPercent">0%</span>
                    <span id="progressPhase">Inicializálás...</span>
                </div>
            </div>
            <div class="phase-indicator" id="phaseStatus">
                Keresési rendszer indítása...<br>
                Téma: <strong>"${escapeHtml(query)}"</strong>
            </div>
            <div class="phases-completed">
                <div class="phase-dot" id="phase1"></div>
                <div class="phase-dot" id="phase2"></div>
                <div class="phase-dot" id="phase3"></div>
                <div class="phase-dot" id="phase4"></div>
            </div>
        </div>`;
    messagesContainer.appendChild(indicatorDiv);
    scrollToBottom();
    simulateResearchProgress();
}

function simulateResearchProgress() {
    const phases = [
        { percent: 25, phase: 'Exa Search futtatása...', status: 'Internetes források keresése...', dot: 'phase1' },
        { percent: 50, phase: 'Gemini Analysis...', status: 'Tartalom elemzése AI-val...', dot: 'phase2' },
        { percent: 75, phase: 'OpenAI Research...', status: 'Mélyebb kutatási elemzés...', dot: 'phase3' },
        { percent: 100, phase: 'Final Synthesis', status: 'Végső jelentés összeállítása...', dot: 'phase4' }
    ];

    let currentPhase = 0;
    const interval = setInterval(() => {
        if (currentPhase < phases.length) {
            const phase = phases[currentPhase];
            const progressFill = getEl('progressFill');
            const progressPercent = getEl('progressPercent');
            const progressPhase = getEl('progressPhase');
            const phaseStatus = getEl('phaseStatus');
            const phaseDot = getEl(phase.dot);

            if (progressFill) progressFill.style.width = phase.percent + '%';
            if (progressPercent) progressPercent.textContent = phase.percent + '%';
            if (progressPhase) progressPhase.textContent = phase.phase;
            if (phaseStatus) phaseStatus.innerHTML = phase.status;
            if (phaseDot) phaseDot.classList.add('completed');

            currentPhase++;
        } else {
            clearInterval(interval);
        }
    }, 2000);
}

function displayResearchCompletion() {
    const researchState = AppState.researchState;
    if (!researchState) return;

    const sourcesCount = researchState.sources ? researchState.sources.length : 0;
    const contentLength = researchState.documentContent ? researchState.documentContent.length : 0;

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message ai-message';
    messageDiv.innerHTML = `
        <div class="research-complete">
            <div class="completion-stats">
                <h3><i class="fas fa-check-circle"></i> Deep Research befejezve!</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <span class="stat-number">${researchState.totalSources || sourcesCount}</span>
                        <span class="stat-label">Forrás</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">${Math.round(contentLength/1000)}k</span>
                        <span class="stat-label">Karakter</span>
                    </div>
                </div>
            </div>
            <div class="document-preview" onclick="showDocumentViewer()">
                <div class="document-icon"><i class="fas fa-file-alt"></i></div>
                <div class="document-info">
                    <h4>Részletes jelentés megtekintése</h4>
                    <p>Kattints a teljes dokumentum megnyitásához</p>
                </div>
            </div>
        </div>`;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function showDocumentViewer() {
    if (AppState.researchState && AppState.researchState.documentContent) {
        // AI üzenet hozzáadása a kutatás eredményével
        addAiMessage(AppState.researchState.documentContent);
    } else {
        addAiMessage("Nincs elérhető dokumentum tartalom.");
    }
}

// Speciális funkciók
const researchCategory = document.createElement('div');
researchCategory.className = 'service-category';
researchCategory.innerHTML = `
    <div class="category-header" onclick="toggleCategory(this)">
        <div class="category-title"><i class="fas fa-search-plus" style="color: var(--secondary)"></i><span>Kutatási Funkciók</span></div>
        <i class="fas fa-chevron-down"></i>
    </div>
    <div class="category-services"><div>
        <div class="service-item" onclick="activateDeepSearch()">
            <div class="service-icon" style="color: var(--secondary);"><i class="fas fa-microscope"></i></div>
            <div class="service-info"><h4>Deep Research</h4><p>Mélyreható kutatási elemzés</p></div>
        </div>
    </div></div>`;

function activateDeepSearch() {
    AppState.isDeepSearchActive = true;
    getEl('deepSearchToggle').classList.add('active');
    messageInput.placeholder = 'Deep Search kutatási téma...';
    messageInput.focus();
    closeSidebar();
}

// --- NATÍV iOS PWA SERVICE WORKER REGISZTRÁCIÓ ---
if ('serviceWorker' in navigator) {
    window.addEventListener('load', async () => {
        try {
            const registration = await navigator.serviceWorker.register('/sw.js');
            console.log('🚀 Service Worker regisztrálva:', registration.scope);

            // Update ellenőrzés
            registration.addEventListener('updatefound', () => {
                const newWorker = registration.installing;
                newWorker.addEventListener('statechange', () => {
                    if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                        // Új verzió elérhető
                        showUpdateNotification();
                    }
                });
            });
        } catch (error) {
            console.log('Service Worker regisztráció sikertelen:', error);
        }
    });
}

// --- iOS NATÍV FUNKCIÓK ---
function setupiOSFeatures() {
    // iOS App Banner elrejtése
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
    });

    // iOS Standalone mód detektálása
    const isStandalone = window.navigator.standalone || 
                       window.matchMedia('(display-mode: standalone)').matches;

    if (isStandalone) {
        document.body.classList.add('ios-standalone');
        console.log('🍎 Natív iOS App módban fut');
    }

    // iOS Viewport fix
    const viewportMeta = document.querySelector('meta[name="viewport"]');
    if (viewportMeta && /iPad|iPhone|iPod/.test(navigator.userAgent)) {
        viewportMeta.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover';
    }

    // iOS Keyboard resize fix
    window.addEventListener('resize', handleiOSKeyboard);

    // iOS Status Bar színezés
    updateiOSStatusBar();
}

function handleiOSKeyboard() {
    if (/iPad|iPhone|iPod/.test(navigator.userAgent)) {
        const windowHeight = window.innerHeight;
        const screenHeight = screen.height;

        if (windowHeight < screenHeight * 0.75) {
            // Billentyűzet megjelent
            document.body.classList.add('keyboard-open');
        } else {
            // Billentyűzet eltűnt
            document.body.classList.remove('keyboard-open');
        }
    }
}

function updateiOSStatusBar() {
    // Dinamikus status bar színezés
    const statusBarMeta = document.querySelector('meta[name="apple-mobile-web-app-status-bar-style"]');
    if (statusBarMeta) {
        // Sötét téma esetén fehér status bar
        const isDark = document.body.classList.contains('dark-theme');
        statusBarMeta.content = isDark ? 'light-content' : 'black-translucent';
    }
}

function showUpdateNotification() {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 20px;
        right: 20px;
        background: var(--glass-strong);
        backdrop-filter: blur(40px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 12px;
        animation: slideDown 0.3s ease;
    `;

    notification.innerHTML = `
        <div style="flex: 1;">
            <div style="font-weight: 600; margin-bottom: 4px;">Új verzió elérhető</div>
            <div style="font-size: 0.8rem; color: var(--text-muted);">Frissítsd az alkalmazást a legújabb funkciókért</div>
        </div>
        <button onclick="reloadApp()" style="
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 0.8rem;
            cursor: pointer;
        ">Frissítés</button>
        <button onclick="this.parentElement.remove()" style="
            background: transparent;
            color: var(--text-muted);
            border: none;
            font-size: 1.2rem;
            cursor: pointer;
            padding: 4px;
        ">×</button>
    `;

    document.body.appendChild(notification);

    setTimeout(() => notification.remove(), 10000);
}

function reloadApp() {
    window.location.reload();
}

// iOS funkciók inicializálása
setupiOSFeatures();
