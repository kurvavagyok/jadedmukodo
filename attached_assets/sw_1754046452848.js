
const CACHE_NAME = 'jaded-ai-v2.5.0';
const STATIC_CACHE = 'jaded-static-v2.5.0';

// Offline támogatás fájljai
const OFFLINE_ASSETS = [
    '/',
    '/components',
    '/static/index.html',
    '/templates/index.html',
    '/templates/components.html',
    'https://unpkg.com/react@18/umd/react.production.min.js',
    'https://unpkg.com/react-dom@18/umd/react-dom.production.min.js',
    'https://cdn.jsdelivr.net/npm/codemirror@5.65.2/lib/codemirror.min.js',
    'https://d3js.org/d3.v7.min.js',
    'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js'
];

// API cache stratégiák
const API_CACHE_STRATEGIES = {
    '/api/services': 'cache-first', // Ritkán változó adatok
    '/api/alphafold3/': 'network-first', // Valós idejű adatok
    '/api/alphagenome/': 'network-first',
    '/api/deep_discovery/': 'network-only' // Mindig friss eredmények
};

// Install event - cache kritikus fájlok
self.addEventListener('install', event => {
    event.waitUntil(
        Promise.all([
            caches.open(STATIC_CACHE).then(cache => {
                return cache.addAll(OFFLINE_ASSETS);
            }),
            caches.open(CACHE_NAME).then(cache => {
                return cache.addAll([
                    '/api/services'
                ]);
            })
        ])
    );
    self.skipWaiting();
});

// Activate event - cleanup régi cache-ek
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME && cacheName !== STATIC_CACHE) {
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    self.clients.claim();
});

// Fetch event - smart caching
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // API kérések kezelése
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(handleApiRequest(request));
        return;
    }

    // Statikus fájlok kezelése
    if (request.destination === 'script' || request.destination === 'style') {
        event.respondWith(handleStaticAsset(request));
        return;
    }

    // Alapértelmezett stratégia
    event.respondWith(handleDefaultRequest(request));
});

// API kérések kezelése stratégia alapján
async function handleApiRequest(request) {
    const url = new URL(request.url);
    const strategy = getApiStrategy(url.pathname);

    switch (strategy) {
        case 'cache-first':
            return cacheFirst(request);
        case 'network-first':
            return networkFirst(request);
        case 'network-only':
            return networkOnly(request);
        default:
            return networkFirst(request);
    }
}

// API stratégia meghatározása
function getApiStrategy(pathname) {
    for (const [pattern, strategy] of Object.entries(API_CACHE_STRATEGIES)) {
        if (pathname.startsWith(pattern)) {
            return strategy;
        }
    }
    return 'network-first';
}

// Cache-first stratégia
async function cacheFirst(request) {
    const cached = await caches.match(request);
    if (cached) {
        return cached;
    }

    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response.clone());
        }
        return response;
    } catch (error) {
        return new Response(JSON.stringify({ error: 'Offline - cached version not available' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Network-first stratégia
async function networkFirst(request) {
    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, response.clone());
        }
        return response;
    } catch (error) {
        const cached = await caches.match(request);
        if (cached) {
            return cached;
        }
        return new Response(JSON.stringify({ error: 'Network error and no cached version' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Network-only stratégia
async function networkOnly(request) {
    try {
        return await fetch(request);
    } catch (error) {
        return new Response(JSON.stringify({ error: 'Network required for this request' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Statikus fájlok kezelése
async function handleStaticAsset(request) {
    const cached = await caches.match(request, { cacheName: STATIC_CACHE });
    if (cached) {
        return cached;
    }

    try {
        const response = await fetch(request);
        if (response.ok) {
            const cache = await caches.open(STATIC_CACHE);
            cache.put(request, response.clone());
        }
        return response;
    } catch (error) {
        return cached || new Response('Asset not available offline', { status: 503 });
    }
}

// Alapértelmezett kérések kezelése
async function handleDefaultRequest(request) {
    try {
        const response = await fetch(request);
        return response;
    } catch (error) {
        // Offline fallback
        if (request.destination === 'document') {
            const cached = await caches.match('/templates/index.html');
            return cached || new Response('Offline', { status: 503 });
        }
        return new Response('Not available offline', { status: 503 });
    }
}

// Background sync a későbbi API kérésekhez
self.addEventListener('sync', event => {
    if (event.tag === 'background-analysis') {
        event.waitUntil(handleBackgroundSync());
    }
});

async function handleBackgroundSync() {
    // Később implementálható: offline módban elmentett kérések újraküldése
    console.log('Background sync triggered');
}

// Push notifications támogatása
self.addEventListener('push', event => {
    if (event.data) {
        const data = event.data.json();
        const options = {
            body: data.body || 'Új elemzési eredmény érkezett',
            icon: '/static/icon-192.png',
            badge: '/static/badge-72.png',
            tag: 'analysis-complete',
            requireInteraction: true,
            actions: [
                {
                    action: 'view',
                    title: 'Megtekintés'
                },
                {
                    action: 'dismiss',
                    title: 'Elvetés'
                }
            ]
        };

        event.waitUntil(
            self.registration.showNotification(data.title || 'JADED AI', options)
        );
    }
});

// Notification click kezelése
self.addEventListener('notificationclick', event => {
    event.notification.close();

    if (event.action === 'view') {
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});
// Service Worker for JADED AI Platform
const CACHE_NAME = 'jaded-v1';
const urlsToCache = [
  '/',
  '/api/services',
  '/static/index.html'
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        if (response) {
          return response;
        }
        return fetch(event.request);
      }
    )
  );
});
