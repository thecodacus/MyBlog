import { graphql } from "gatsby"
import React from "react"
import Layout from "../../components/Layout"
import PageHeroName from "../../components/PageHeroName"
import PostsGrid from "../../components/PostsGrid"
import * as styles from "../../styles/posts.module.scss"
export default function Posts({ data }) {
	console.log(data)
	const posts = data.posts.nodes
	let categories = Array.from(new Set(posts.map(post => post.frontmatter.category)))
	let postGrids = categories.map((category, index) => (
		<div key={index}>
			<div className={styles.header}>{category}</div>
			<PostsGrid posts={posts.filter(post => post.frontmatter.category === category)}></PostsGrid>
		</div>
	))
	return (
		<Layout>
			<Seo title="Posts" />
			<PageHeroName>Posts</PageHeroName>
			<section>{postGrids}</section>
		</Layout>
	)
}

export const query = graphql`
	query AllPosts {
		posts: allMarkdownRemark(sort: { fields: frontmatter___date, order: DESC }) {
			nodes {
				fields {
					slug
				}
				excerpt
				frontmatter {
					title
					category
					date
					featuredImage {
						publicURL
						childImageSharp {
							gatsbyImageData(aspectRatio: 1.5, layout: FULL_WIDTH)
						}
					}
				}
				id
			}
		}
	}
`
