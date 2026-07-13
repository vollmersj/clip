import type { ComponentType } from 'react';
import { Ch1BigIdea } from './Ch1BigIdea';
import { Ch2Encoders } from './Ch2Encoders';
import { Ch3SharedSpace } from './Ch3SharedSpace';
import { Ch4SimilarityMatrix } from './Ch4SimilarityMatrix';
import { Ch5ContrastiveLoss } from './Ch5ContrastiveLoss';
import { Ch6TrainingSim } from './Ch6TrainingSim';
import { Ch7ZeroShot } from './Ch7ZeroShot';
import { Ch8Retrieval } from './Ch8Retrieval';
import { Ch9Playground } from './Ch9Playground';
import { Ch10Recap } from './Ch10Recap';

export interface Chapter {
  slug: string;
  /** Short name for nav buttons and the header. */
  short: string;
  /** Full title shown in the table of contents. */
  title: string;
  component: ComponentType;
}

export const chapters: Chapter[] = [
  { slug: 'big-idea', short: 'The Big Idea', title: 'The Big Idea: Bridging Vision and Language', component: Ch1BigIdea },
  { slug: 'encoders', short: 'Two Encoders', title: 'Two Encoders, One Goal', component: Ch2Encoders },
  { slug: 'shared-space', short: 'The Shared Space', title: 'A Shared Embedding Space', component: Ch3SharedSpace },
  { slug: 'similarity-matrix', short: 'Similarity Matrix', title: 'The Similarity Matrix', component: Ch4SimilarityMatrix },
  { slug: 'contrastive-loss', short: 'Contrastive Loss', title: 'The Contrastive Loss', component: Ch5ContrastiveLoss },
  { slug: 'train-it', short: 'Train It Yourself', title: 'Train It Yourself: A Tiny CLIP', component: Ch6TrainingSim },
  { slug: 'zero-shot', short: 'Zero-Shot Classification', title: 'Zero-Shot Classification', component: Ch7ZeroShot },
  { slug: 'retrieval', short: 'Retrieval', title: 'Retrieval: Search in Both Directions', component: Ch8Retrieval },
  { slug: 'playground', short: 'Playground', title: 'Playground: Run CLIP Live', component: Ch9Playground },
  { slug: 'recap', short: 'Recap & Quiz', title: 'Recap, Pseudocode & Quiz', component: Ch10Recap },
];
