import { graphql, Link } from "gatsby"
import React from "react"
import Img from "gatsby-image"
import Layout from "../../components/Layout"
import * as styles from "../../styles/posts.module.css"
export default function Posts({ data }) {
	console.log(data)
	const posts = data.posts.nodes
	return (
		<Layout>
			<section>
				<div>Posts Works</div>
				<div className={styles.posts}>
					{posts.map(post => (
						<Link to={"/posts/" + post.frontmatter.slug} key={post.id}>
							<div>
								<Img fluid={post.frontmatter.featuredImage.childImageSharp.fluid} />
								<h3>{post.frontmatter.title}</h3>
								<p>{post.frontmatter.category}</p>
							</div>
						</Link>
					))}
				</div>
			</section>
		</Layout>
	)
}

export const query = graphql`
	query AllPosts {
		posts: allMarkdownRemark(sort: { fields: frontmatter___date, order: DESC }) {
			nodes {
				frontmatter {
					title
					slug
					category
					date
					featuredImage {
						childImageSharp {
							fluid {
								...GatsbyImageSharpFluid
							}
						}
					}
				}
				id
			}
		}
	}
`
