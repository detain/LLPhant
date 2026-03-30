<?php

namespace LLPhant\Embeddings\VectorStores\Pinecone;

use Probots\Pinecone\Client as ProbotsClient;

class PineconeClient
{
    public ProbotsClient $client;

    /**
     * @param  string  $host  The full index host url (e.g., index-name-project.svc.environment.pinecone.io)
     * @param  string  $apiKey  The Pinecone API Key
     */
    public function __construct(
        public string $host,
        public string $apiKey
    ) {
        $this->client = new ProbotsClient($apiKey);
        $this->client->setIndexHost('https://'.$host);
    }

    /**
     * @return array<string, mixed>
     */
    public function describeIndexStats(): array
    {
        $response = $this->client->data()->vectors()->stats();

        if (! $response->successful()) {
            throw new \Exception('Failed to fetch index stats: '.$response->body());
        }

        return $response->json();
    }

    /**
     * @param  array<int, array<string, mixed>>  $vectors
     */
    public function upsert(array $vectors, string $namespace = ''): bool
    {
        $response = $this->client->data()->vectors()->upsert(
            vectors: $vectors,
            namespace: $namespace
        );

        return $response->successful();
    }

    /**
     * @param  float[]  $vector
     * @param  array<string, mixed>|null  $filter
     * @return array<mixed>
     */
    public function query(
        array $vector,
        int $topK,
        ?array $filter = null,
        string $namespace = '',
        bool $includeMetadata = true,
        bool $includeValues = false
    ): array {
        $response = $this->client->data()->vectors()->query(
            vector: $vector,
            namespace: $namespace,
            filter: $filter ?? [],
            topK: $topK,
            includeMetadata: $includeMetadata,
            includeValues: $includeValues
        );

        if (! $response->successful()) {
            throw new \Exception('Pinecone Query Failed: '.$response->body());
        }

        return $response->json()['matches'] ?? [];
    }
}
