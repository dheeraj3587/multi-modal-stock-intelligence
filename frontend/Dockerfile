# ============================================================
# Frontend Dockerfile for Next.js Application
# ============================================================
# This Dockerfile uses Next.js standalone output mode for optimal production builds.
# 
# IMPORTANT: Your next.config.js must include:
#   output: 'standalone'
#
# This configuration generates a .next/standalone directory with a minimal
# server.js entry point that includes only the necessary dependencies.
#
# If you prefer the standard Next.js server instead, you can:
# 1. Remove the "output: 'standalone'" from next.config.js
# 2. In the runner stage, copy .next/, public/, and node_modules/
# 3. Change CMD to: ["npm", "start"] or ["node", "node_modules/next/dist/bin/next", "start"]
# ============================================================

# Stage 1 - Dependencies: Install node modules
FROM node:18-alpine AS dependencies

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies with frozen lockfile for reproducibility
RUN npm ci --frozen-lockfile

# Stage 2 - Builder: Build the Next.js application
FROM node:18-alpine AS builder

WORKDIR /app

# Copy node_modules from dependencies stage
COPY --from=dependencies /app/node_modules ./node_modules

# Copy source code
COPY . .

# Set build-time environment variables if needed
# COPY .env.local .env.local

# Build Next.js application
RUN npm run build

# Stage 3 - Runner: Production runtime
FROM node:18-alpine AS runner

WORKDIR /app

# Set production environment
ENV NODE_ENV=production

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy necessary files from builder
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Set ownership to non-root user
RUN chown -R nextjs:nodejs /app

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Set port environment variable
ENV PORT 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/api/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})" || exit 1

# Start Next.js application
CMD ["node", "server.js"]

