# RMG frontend (Next.js + React + Tailwind). Multi-stage → small standalone image.
#
#   docker build -f frontend/Dockerfile -t rmg-frontend ./frontend
#
# NEXT_PUBLIC_* values are inlined at BUILD time (the browser calls the backend
# directly), so the API base must be a URL the browser can reach.
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json ./
RUN npm install

FROM node:20-slim AS builder
WORKDIR /app
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000
# Next.js `output: standalone` emits a self-contained server bundle.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
